import ctypes
import functools
from winappdbg import compat
import sys
def RaiseIfNotErrorSuccess(result, func=None, arguments=()):
    """
    Error checking for Win32 Registry API calls.

    The function is assumed to return a Win32 error code. If the code is not
    C{ERROR_SUCCESS} then a C{WindowsError} exception is raised.
    """
    if result != ERROR_SUCCESS:
        raise ctypes.WinError(result)
    return result