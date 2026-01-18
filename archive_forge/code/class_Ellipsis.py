import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
class Ellipsis(Constant, metaclass=_ABC):
    _fields = ()

    def __new__(cls, *args, **kwargs):
        if cls is Ellipsis:
            return Constant(..., *args, **kwargs)
        return Constant.__new__(cls, *args, **kwargs)