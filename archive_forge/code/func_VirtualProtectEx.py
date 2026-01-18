import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def VirtualProtectEx(hProcess, lpAddress, dwSize, flNewProtect=PAGE_EXECUTE_READWRITE):
    _VirtualProtectEx = windll.kernel32.VirtualProtectEx
    _VirtualProtectEx.argtypes = [HANDLE, LPVOID, SIZE_T, DWORD, PDWORD]
    _VirtualProtectEx.restype = bool
    _VirtualProtectEx.errcheck = RaiseIfZero
    flOldProtect = DWORD(0)
    _VirtualProtectEx(hProcess, lpAddress, dwSize, flNewProtect, byref(flOldProtect))
    return flOldProtect.value