import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def _is_infix_compact(k):
    global _infix_compact_map
    return _infix_compact_map.get(k, False)