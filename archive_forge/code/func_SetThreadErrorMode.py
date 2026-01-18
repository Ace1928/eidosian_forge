import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetThreadErrorMode(dwNewMode):
    _SetThreadErrorMode = windll.kernel32.SetThreadErrorMode
    _SetThreadErrorMode.argtypes = [DWORD, LPDWORD]
    _SetThreadErrorMode.restype = BOOL
    _SetThreadErrorMode.errcheck = RaiseIfZero
    old = DWORD(0)
    _SetThreadErrorMode(dwErrCode, byref(old))
    return old.value