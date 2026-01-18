import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def OpenMutexW(dwDesiredAccess=MUTEX_ALL_ACCESS, bInitialOwner=True, lpName=None):
    _OpenMutexW = windll.kernel32.OpenMutexW
    _OpenMutexW.argtypes = [DWORD, BOOL, LPWSTR]
    _OpenMutexW.restype = HANDLE
    _OpenMutexW.errcheck = RaiseIfZero
    return Handle(_OpenMutexW(lpMutexAttributes, bInitialOwner, lpName))