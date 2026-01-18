import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def DebugBreakProcess(hProcess):
    _DebugBreakProcess = windll.kernel32.DebugBreakProcess
    _DebugBreakProcess.argtypes = [HANDLE]
    _DebugBreakProcess.restype = bool
    _DebugBreakProcess.errcheck = RaiseIfZero
    _DebugBreakProcess(hProcess)