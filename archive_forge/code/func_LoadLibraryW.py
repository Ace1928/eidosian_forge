import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def LoadLibraryW(pszLibrary):
    _LoadLibraryW = windll.kernel32.LoadLibraryW
    _LoadLibraryW.argtypes = [LPWSTR]
    _LoadLibraryW.restype = HMODULE
    hModule = _LoadLibraryW(pszLibrary)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule