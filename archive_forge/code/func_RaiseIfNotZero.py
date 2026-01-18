import ctypes
import functools
from winappdbg import compat
import sys
def RaiseIfNotZero(result, func=None, arguments=()):
    """
    Error checking for some odd Win32 API calls.

    The function is assumed to return an integer, which is zero on success.
    If the return value is nonzero the C{WindowsError} exception is raised.

    This is mostly useful for free() like functions, where the return value is
    the pointer to the memory block on failure or a C{NULL} pointer on success.
    """
    if result:
        raise ctypes.WinError()
    return result