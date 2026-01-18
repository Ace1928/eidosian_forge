from __future__ import annotations
import enum
from .types import Type, Bool, Uint
def _order_uint_uint(left: Uint, right: Uint, /) -> Ordering:
    if left.width < right.width:
        return Ordering.LESS
    if left.width == right.width:
        return Ordering.EQUAL
    return Ordering.GREATER