import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def WaitForMultipleObjectsEx(handles, bWaitAll=False, dwMilliseconds=INFINITE, bAlertable=True):
    _WaitForMultipleObjectsEx = windll.kernel32.WaitForMultipleObjectsEx
    _WaitForMultipleObjectsEx.argtypes = [DWORD, POINTER(HANDLE), BOOL, DWORD]
    _WaitForMultipleObjectsEx.restype = DWORD
    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    nCount = len(handles)
    lpHandlesType = HANDLE * nCount
    lpHandles = lpHandlesType(*handles)
    if dwMilliseconds != INFINITE:
        r = _WaitForMultipleObjectsEx(byref(lpHandles), bool(bWaitAll), dwMilliseconds, bool(bAlertable))
        if r == WAIT_FAILED:
            raise ctypes.WinError()
    else:
        while 1:
            r = _WaitForMultipleObjectsEx(byref(lpHandles), bool(bWaitAll), 100, bool(bAlertable))
            if r == WAIT_FAILED:
                raise ctypes.WinError()
            if r != WAIT_TIMEOUT:
                break
    return r