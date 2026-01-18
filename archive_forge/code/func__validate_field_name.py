from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_field_name(self, name, val):
    try:
        assert val in self._field_names or val is None
    except AssertionError:
        msg = f'Invalid field name: {val}'
        raise ValueError(msg)