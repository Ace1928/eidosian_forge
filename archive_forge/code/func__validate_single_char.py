from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_single_char(self, name, val):
    try:
        assert _str_block_width(val) == 1
    except AssertionError:
        msg = f'Invalid value for {name}. Must be a string of length 1.'
        raise ValueError(msg)