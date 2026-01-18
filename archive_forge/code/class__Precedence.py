import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
@_simple_enum(IntEnum)
class _Precedence:
    """Precedence table that originated from python grammar."""
    NAMED_EXPR = auto()
    TUPLE = auto()
    YIELD = auto()
    TEST = auto()
    OR = auto()
    AND = auto()
    NOT = auto()
    CMP = auto()
    EXPR = auto()
    BOR = EXPR
    BXOR = auto()
    BAND = auto()
    SHIFT = auto()
    ARITH = auto()
    TERM = auto()
    FACTOR = auto()
    POWER = auto()
    AWAIT = auto()
    ATOM = auto()

    def next(self):
        try:
            return self.__class__(self + 1)
        except ValueError:
            return self