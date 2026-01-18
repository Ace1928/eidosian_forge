import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CreateMutexA(lpMutexAttributes=None, bInitialOwner=True, lpName=None):
    _CreateMutexA = windll.kernel32.CreateMutexA
    _CreateMutexA.argtypes = [LPVOID, BOOL, LPSTR]
    _CreateMutexA.restype = HANDLE
    _CreateMutexA.errcheck = RaiseIfZero
    return Handle(_CreateMutexA(lpMutexAttributes, bInitialOwner, lpName))