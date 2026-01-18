import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CloseHandle(hHandle):
    if isinstance(hHandle, Handle):
        hHandle.close()
    else:
        _CloseHandle = windll.kernel32.CloseHandle
        _CloseHandle.argtypes = [HANDLE]
        _CloseHandle.restype = bool
        _CloseHandle.errcheck = RaiseIfZero
        _CloseHandle(hHandle)