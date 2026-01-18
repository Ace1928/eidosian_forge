import sys
from _ast import *
from contextlib import contextmanager, nullcontext
from enum import IntEnum, auto, _simple_enum
class NameConstant(Constant, metaclass=_ABC):
    __new__ = _new