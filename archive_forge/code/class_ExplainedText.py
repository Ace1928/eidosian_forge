from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
class ExplainedText(NamedTuple):
    """
    The return type from ftfy's functions that provide an "explanation" of which
    steps it applied to fix the text, such as :func:`fix_and_explain()`.

    When the 'explain' option is disabled, these functions return the same
    type, but the `explanation` will be None.
    """
    text: str
    explanation: Optional[List[ExplanationStep]]