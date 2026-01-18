from __future__ import annotations
from ..keys import Keys
def _get_reverse_ansi_sequences() -> dict[Keys, str]:
    """
    Create a dictionary that maps prompt_toolkit keys back to the VT100 escape
    sequences.
    """
    result: dict[Keys, str] = {}
    for sequence, key in ANSI_SEQUENCES.items():
        if not isinstance(key, tuple):
            if key not in result:
                result[key] = sequence
    return result