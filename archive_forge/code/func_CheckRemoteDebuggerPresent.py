import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CheckRemoteDebuggerPresent(hProcess):
    _CheckRemoteDebuggerPresent = windll.kernel32.CheckRemoteDebuggerPresent
    _CheckRemoteDebuggerPresent.argtypes = [HANDLE, PBOOL]
    _CheckRemoteDebuggerPresent.restype = bool
    _CheckRemoteDebuggerPresent.errcheck = RaiseIfZero
    pbDebuggerPresent = BOOL(0)
    _CheckRemoteDebuggerPresent(hProcess, byref(pbDebuggerPresent))
    return bool(pbDebuggerPresent.value)