from __future__ import annotations
import enum
from .types import Type, Bool, Uint
def _order_bool_bool(_a: Bool, _b: Bool, /) -> Ordering:
    return Ordering.EQUAL