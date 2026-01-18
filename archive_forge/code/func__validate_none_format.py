from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_none_format(self, val):
    try:
        if val is not None:
            assert isinstance(val, str)
    except AssertionError:
        msg = 'Replacement for None value must be a string if being supplied.'
        raise TypeError(msg)