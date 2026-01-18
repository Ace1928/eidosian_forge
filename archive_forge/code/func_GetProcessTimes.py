import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetProcessTimes(hProcess=None):
    _GetProcessTimes = windll.kernel32.GetProcessTimes
    _GetProcessTimes.argtypes = [HANDLE, LPFILETIME, LPFILETIME, LPFILETIME, LPFILETIME]
    _GetProcessTimes.restype = bool
    _GetProcessTimes.errcheck = RaiseIfZero
    if hProcess is None:
        hProcess = GetCurrentProcess()
    CreationTime = FILETIME()
    ExitTime = FILETIME()
    KernelTime = FILETIME()
    UserTime = FILETIME()
    _GetProcessTimes(hProcess, byref(CreationTime), byref(ExitTime), byref(KernelTime), byref(UserTime))
    return (CreationTime, ExitTime, KernelTime, UserTime)