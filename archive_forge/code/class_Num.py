import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
class Num(Constant, metaclass=_ABC):
    _fields = ('n',)
    __new__ = _new