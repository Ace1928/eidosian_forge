import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetModuleHandleW(lpModuleName):
    _GetModuleHandleW = windll.kernel32.GetModuleHandleW
    _GetModuleHandleW.argtypes = [LPWSTR]
    _GetModuleHandleW.restype = HMODULE
    hModule = _GetModuleHandleW(lpModuleName)
    if hModule == NULL:
        raise ctypes.WinError()
    return hModule