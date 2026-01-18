import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def RaiseIfLastError(result, func=None, arguments=()):
    """
    Error checking for Win32 API calls with no error-specific return value.

    Regardless of the return value, the function calls GetLastError(). If the
    code is not C{ERROR_SUCCESS} then a C{WindowsError} exception is raised.

    For this to work, the user MUST call SetLastError(ERROR_SUCCESS) prior to
    calling the API. Otherwise an exception may be raised even on success,
    since most API calls don't clear the error status code.
    """
    code = GetLastError()
    if code != ERROR_SUCCESS:
        raise ctypes.WinError(code)
    return result