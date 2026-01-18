import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def DebugActiveProcessStop(dwProcessId):
    _DebugActiveProcessStop = windll.kernel32.DebugActiveProcessStop
    _DebugActiveProcessStop.argtypes = [DWORD]
    _DebugActiveProcessStop.restype = bool
    _DebugActiveProcessStop.errcheck = RaiseIfZero
    _DebugActiveProcessStop(dwProcessId)