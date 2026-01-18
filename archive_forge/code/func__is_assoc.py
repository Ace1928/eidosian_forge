import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _is_assoc(k):
    return k in _ASSOC_OPS