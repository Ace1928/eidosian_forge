import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetProcessVersion(ProcessId):
    _GetProcessVersion = windll.kernel32.GetProcessVersion
    _GetProcessVersion.argtypes = [DWORD]
    _GetProcessVersion.restype = DWORD
    retval = _GetProcessVersion(ProcessId)
    if retval == 0:
        raise ctypes.WinError()
    return retval