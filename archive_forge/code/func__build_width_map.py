from __future__ import annotations
import html
import itertools
import re
import unicodedata
def _build_width_map():
    """
    Build a translate mapping that replaces halfwidth and fullwidth forms
    with their standard-width forms.
    """
    width_map = {12288: ' '}
    for i in range(65281, 65520):
        char = chr(i)
        alternate = unicodedata.normalize('NFKC', char)
        if alternate != char:
            width_map[i] = alternate
    return width_map