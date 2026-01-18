from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_true_or_false(self, name, val):
    try:
        assert val in (True, False)
    except AssertionError:
        msg = f'Invalid value for {name}. Must be True or False.'
        raise ValueError(msg)