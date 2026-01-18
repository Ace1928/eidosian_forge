from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_header_style(self, val):
    try:
        assert val in ('cap', 'title', 'upper', 'lower', None)
    except AssertionError:
        msg = 'Invalid header style, use cap, title, upper, lower or None'
        raise ValueError(msg)