from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
def _uri_encode_value(self, value: str) -> str:
    """Encode a value into uri encoding."""
    return self._encode(value, Charset.UNRESERVED + Charset.RESERVED, True)