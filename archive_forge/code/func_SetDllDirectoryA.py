import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetDllDirectoryA(lpPathName=None):
    _SetDllDirectoryA = windll.kernel32.SetDllDirectoryA
    _SetDllDirectoryA.argytpes = [LPSTR]
    _SetDllDirectoryA.restype = bool
    _SetDllDirectoryA.errcheck = RaiseIfZero
    _SetDllDirectoryA(lpPathName)