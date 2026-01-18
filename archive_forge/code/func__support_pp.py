import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _support_pp(a):
    return isinstance(a, z3.Z3PPObject) or isinstance(a, list) or isinstance(a, tuple)