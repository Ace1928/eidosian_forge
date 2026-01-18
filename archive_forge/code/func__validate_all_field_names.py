from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_all_field_names(self, name, val):
    try:
        for x in val:
            self._validate_field_name(name, x)
    except AssertionError:
        msg = 'Fields must be a sequence of field names'
        raise ValueError(msg)