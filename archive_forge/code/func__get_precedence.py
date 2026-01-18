import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _get_precedence(k):
    global _z3_precedence
    return _z3_precedence.get(k, 100000)