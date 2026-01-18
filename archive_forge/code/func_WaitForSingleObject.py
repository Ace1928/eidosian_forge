import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def WaitForSingleObject(hHandle, dwMilliseconds=INFINITE):
    _WaitForSingleObject = windll.kernel32.WaitForSingleObject
    _WaitForSingleObject.argtypes = [HANDLE, DWORD]
    _WaitForSingleObject.restype = DWORD
    if not dwMilliseconds and dwMilliseconds != 0:
        dwMilliseconds = INFINITE
    if dwMilliseconds != INFINITE:
        r = _WaitForSingleObject(hHandle, dwMilliseconds)
        if r == WAIT_FAILED:
            raise ctypes.WinError()
    else:
        while 1:
            r = _WaitForSingleObject(hHandle, 100)
            if r == WAIT_FAILED:
                raise ctypes.WinError()
            if r != WAIT_TIMEOUT:
                break
    return r