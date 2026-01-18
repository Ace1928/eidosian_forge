from __future__ import annotations
import collections
from typing import Any, TYPE_CHECKING, cast
from .charset import Charset
from .variable import Variable
class ReservedExpansion(ExpressionExpansion):
    """
    Reserved Expansion {+var}.

    https://tools.ietf.org/html/rfc6570#section-3.2.3
    """
    operator = '+'
    partial_operator = ',+'

    def __init__(self, variables: str) -> None:
        super().__init__(variables[1:])

    def _uri_encode_value(self, value: str) -> str:
        """Encode a value into uri encoding."""
        return self._encode(value, Charset.UNRESERVED + Charset.RESERVED, True)