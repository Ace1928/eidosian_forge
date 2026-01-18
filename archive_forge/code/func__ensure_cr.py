from __future__ import annotations
import functools
import re
import typing as ty
import warnings
def _ensure_cr(text: str) -> str:
    """Remove trailing whitespace and add carriage return

    Ensures that `text` always ends with a carriage return
    """
    return text.rstrip() + '\n'