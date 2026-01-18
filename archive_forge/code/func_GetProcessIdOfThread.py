import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetProcessIdOfThread(hThread):
    _GetProcessIdOfThread = windll.kernel32.GetProcessIdOfThread
    _GetProcessIdOfThread.argtypes = [HANDLE]
    _GetProcessIdOfThread.restype = DWORD
    dwProcessId = _GetProcessIdOfThread(hThread)
    if dwProcessId == 0:
        raise ctypes.WinError()
    return dwProcessId