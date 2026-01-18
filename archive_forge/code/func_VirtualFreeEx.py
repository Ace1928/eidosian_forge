import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def VirtualFreeEx(hProcess, lpAddress, dwSize=0, dwFreeType=MEM_RELEASE):
    _VirtualFreeEx = windll.kernel32.VirtualFreeEx
    _VirtualFreeEx.argtypes = [HANDLE, LPVOID, SIZE_T, DWORD]
    _VirtualFreeEx.restype = bool
    _VirtualFreeEx.errcheck = RaiseIfZero
    _VirtualFreeEx(hProcess, lpAddress, dwSize, dwFreeType)