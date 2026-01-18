from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_valign(self, val):
    try:
        assert val in ['t', 'm', 'b', None]
    except AssertionError:
        msg = f'Alignment {val} is invalid, use t, m, b or None'
        raise ValueError(msg)