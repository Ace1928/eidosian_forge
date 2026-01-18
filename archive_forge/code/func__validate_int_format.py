from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_int_format(self, name, val):
    if val == '':
        return
    try:
        assert isinstance(val, str)
        assert val.isdigit()
    except AssertionError:
        msg = f'Invalid value for {name}. Must be an integer format string.'
        raise ValueError(msg)