import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
class ExtSlice(slice):
    """Deprecated AST node class. Use ast.Tuple instead."""

    def __new__(cls, dims=(), **kwargs):
        return Tuple(list(dims), Load(), **kwargs)