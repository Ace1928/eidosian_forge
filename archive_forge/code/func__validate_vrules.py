from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_vrules(self, name, val):
    try:
        assert val in (ALL, FRAME, NONE)
    except AssertionError:
        msg = f'Invalid value for {name}. Must be ALL, FRAME, or NONE.'
        raise ValueError(msg)