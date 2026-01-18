from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
def _errcheck_dword(value, func, args):
    if value == 4294967295:
        raise ctypes.WinError
    return args