import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _z3_assert(cond, msg):
    if not cond:
        raise Z3Exception(msg)