from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
def _uri_encode_name(self, name: str | int) -> str:
    """Encode a variable name into uri encoding."""
    return self._encode(str(name), Charset.UNRESERVED + Charset.RESERVED, True) if name else ''