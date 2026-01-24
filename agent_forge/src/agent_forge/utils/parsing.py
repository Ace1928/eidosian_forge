from typing import Any, Dict, Optional
import json
from eidosian_core import eidosian

@eidosian()
def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object from a given text, even if it's wrapped in Markdown code blocks.

    Args:
        text: The string potentially containing a JSON object.

    Returns:
        The parsed JSON object as a dictionary, or None if no valid JSON is found.
    """
    try:
        # Try to extract JSON from a Markdown code block
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            # Fallback for generic code blocks
            json_str = text.split("```")[1].split("```")[0].strip()
        else:
            json_str = text.strip() # Assume plain JSON

        return json.loads(json_str)
    except (json.JSONDecodeError, IndexError):
        return None
