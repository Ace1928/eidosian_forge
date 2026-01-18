import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def WaitForSingleObjectEx(hHandle, dwMilliseconds=INFINITE, bAlertable=True):
    _WaitForSingleObjectEx = windll.kernel32.WaitForSingleObjectEx
    _WaitForSingleObjectEx.argtypes = [HANDLE, DWORD, BOOL]
    _WaitForSingleObjectEx.restype = DWORD
    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    if dwMilliseconds != INFINITE:
        r = _WaitForSingleObjectEx(hHandle, dwMilliseconds, bool(bAlertable))
        if r == WAIT_FAILED:
            raise ctypes.WinError()
    else:
        while 1:
            r = _WaitForSingleObjectEx(hHandle, 100, bool(bAlertable))
            if r == WAIT_FAILED:
                raise ctypes.WinError()
            if r != WAIT_TIMEOUT:
                break
    return r