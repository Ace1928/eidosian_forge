import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def FreeLibrary(hModule):
    _FreeLibrary = windll.kernel32.FreeLibrary
    _FreeLibrary.argtypes = [HMODULE]
    _FreeLibrary.restype = bool
    _FreeLibrary.errcheck = RaiseIfZero
    _FreeLibrary(hModule)