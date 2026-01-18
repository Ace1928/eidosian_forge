import atexit
import sys, os
import contextlib
import ctypes
from .z3types import *
from .z3consts import *
def _str_to_bytes(s):
    if isinstance(s, str):
        enc = sys.getdefaultencoding()
        return s.encode(enc if enc != None else 'latin-1')
    else:
        return s