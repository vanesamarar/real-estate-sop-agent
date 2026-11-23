import os
import time
from google import genai
from google.genai import types

MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.1-flash-lite",
]

def load_file(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()

def generate_with_fallback(client, prompt: str, system_instruction: str) -> str:
    last_error = None
    for model_name in MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                ),
            )
            return response.text or ""
        except Exception as e:
            last_error = e
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"[{model_name} busy, trying next model...]")
                time.sleep(1)
                continue
            raise
    raise RuntimeError(f"All models temporarily unavailable: {last_error}")

def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return

    client = genai.Client(api_key=api_key)
    sop_text = load_file("sop_listings.txt")
    email_text = load_file("agent_email.txt")

    system_instruction = (
        "You are a strict, reliable Real Estate Operations Assistant for a Bellevue, WA brokerage. "
        "Base every decision strictly on the provided SOP document and incoming email. "
        "Never guess or invent missing data. Cite the exact SOP Rule number (e.g., [Rule 2.2]) for every action."
    )

    prompt = f"""
Analyze the incoming agent email against our brokerage SOP and produce a clean operations handoff report with these exact four sections:

1. STRUCTURED PROPERTY DATA (Key-Value list; mark any missing required fields as MISSING)
2. COMPLIANCE & PAPERWORK CHECKLIST (Bullet points with exact SOP rule citations)
3. SALESFORCE & BLOCKERS (What can be updated now vs. what is on hold)
4. READY-TO-SEND VENDOR EMAIL DRAFTS (Only draft vendor emails permitted by the SOP rules)

--- SOP DOCUMENT ---
{sop_text}

--- INCOMING AGENT EMAIL ---
{email_text}
"""

    print("\nRunning Listing Intake & SOP Agent...\n" + "=" * 50)
    output_text = generate_with_fallback(client, prompt, system_instruction)
    print(output_text)

    with open("intake_report_output.txt", "w", encoding="utf-8") as out:
        out.write(output_text)
    print("\n" + "=" * 50)
    print("Saved report to intake_report_output.txt")

    print("\n--- SOP Q&A Mode (type 'exit' to quit) ---")
    while True:
        question = input("\nAsk an SOP question: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break

        qa_prompt = f"""
Answer the user's question using ONLY the SOP document and the current listing email below. Cite rule numbers.

--- SOP DOCUMENT ---
{sop_text}

--- CURRENT LISTING EMAIL ---
{email_text}

Question: {question}
"""
        answer = generate_with_fallback(client, qa_prompt, system_instruction)
        print(f"\nAgent: {answer}")

if __name__ == "__main__":
    main()
