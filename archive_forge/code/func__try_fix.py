from __future__ import annotations
import unicodedata
import warnings
from typing import (
from ftfy import bad_codecs
from ftfy import chardata, fixes
from ftfy.badness import is_bad
from ftfy.formatting import display_ljust
def _try_fix(fixer_name: str, text: str, config: TextFixerConfig, steps: Optional[List[ExplanationStep]]) -> str:
    """
    A helper function used across several 'fixer' steps, deciding whether to
    apply the fix and whether to record the fix in `steps`.
    """
    if getattr(config, fixer_name):
        fixer = FIXERS[fixer_name]
        fixed = fixer(text)
        if steps is not None and fixed != text:
            steps.append(ExplanationStep('apply', fixer_name))
        return cast(str, fixed)
    return text