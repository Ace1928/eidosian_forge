from __future__ import annotations
import io
import re
from html.parser import HTMLParser
from typing import Any
def _validate_field_names(self, val):
    if self._field_names:
        try:
            assert len(val) == len(self._field_names)
        except AssertionError:
            msg = f'Field name list has incorrect number of values, (actual) {len(val)}!={len(self._field_names)} (expected)'
            raise ValueError(msg)
    if self._rows:
        try:
            assert len(val) == len(self._rows[0])
        except AssertionError:
            msg = f'Field name list has incorrect number of values, (actual) {len(val)}!={len(self._rows[0])} (expected)'
            raise ValueError(msg)
    try:
        assert len(val) == len(set(val))
    except AssertionError:
        msg = 'Field names must be unique'
        raise ValueError(msg)