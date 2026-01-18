from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
def _errcheck_bool(value, func, args):
    if not value:
        raise ctypes.WinError
    return args