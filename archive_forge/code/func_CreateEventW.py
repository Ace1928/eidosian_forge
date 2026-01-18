import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CreateEventW(lpMutexAttributes=None, bManualReset=False, bInitialState=False, lpName=None):
    _CreateEventW = windll.kernel32.CreateEventW
    _CreateEventW.argtypes = [LPVOID, BOOL, BOOL, LPWSTR]
    _CreateEventW.restype = HANDLE
    _CreateEventW.errcheck = RaiseIfZero
    return Handle(_CreateEventW(lpMutexAttributes, bManualReset, bInitialState, lpName))