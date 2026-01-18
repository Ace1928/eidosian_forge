from __future__ import annotations
import html
import itertools
import re
import unicodedata
def _build_control_char_mapping():
    """
    Build a translate mapping that strips likely-unintended control characters.
    See :func:`ftfy.fixes.remove_control_chars` for a description of these
    codepoint ranges and why they should be removed.
    """
    control_chars: dict[int, None] = {}
    for i in itertools.chain(range(0, 9), [11], range(14, 32), [127], range(8298, 8304), [65279], range(65529, 65533)):
        control_chars[i] = None
    return control_chars