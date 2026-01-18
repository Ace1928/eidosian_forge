import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetExitCodeProcess(hProcess):
    _GetExitCodeProcess = windll.kernel32.GetExitCodeProcess
    _GetExitCodeProcess.argtypes = [HANDLE]
    _GetExitCodeProcess.restype = bool
    _GetExitCodeProcess.errcheck = RaiseIfZero
    lpExitCode = DWORD(0)
    _GetExitCodeProcess(hProcess, byref(lpExitCode))
    return lpExitCode.value