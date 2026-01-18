from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
def _errcheck_handle(value, func, args):
    if not value:
        raise ctypes.WinError
    if value == INVALID_HANDLE_VALUE:
        raise ctypes.WinError
    return args