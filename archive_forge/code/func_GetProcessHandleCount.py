import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetProcessHandleCount(hProcess):
    _GetProcessHandleCount = windll.kernel32.GetProcessHandleCount
    _GetProcessHandleCount.argtypes = [HANDLE, PDWORD]
    _GetProcessHandleCount.restype = DWORD
    _GetProcessHandleCount.errcheck = RaiseIfZero
    pdwHandleCount = DWORD(0)
    _GetProcessHandleCount(hProcess, byref(pdwHandleCount))
    return pdwHandleCount.value