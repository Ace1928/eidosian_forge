import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _is_add(k):
    return k == Z3_OP_ADD or k == Z3_OP_BADD or k == Z3_OP_FPA_ADD