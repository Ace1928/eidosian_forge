import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def OpenEventA(dwDesiredAccess=EVENT_ALL_ACCESS, bInheritHandle=False, lpName=None):
    _OpenEventA = windll.kernel32.OpenEventA
    _OpenEventA.argtypes = [DWORD, BOOL, LPSTR]
    _OpenEventA.restype = HANDLE
    _OpenEventA.errcheck = RaiseIfZero
    return Handle(_OpenEventA(dwDesiredAccess, bInheritHandle, lpName))