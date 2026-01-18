import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetThreadId(hThread):
    _GetThreadId = windll.kernel32._GetThreadId
    _GetThreadId.argtypes = [HANDLE]
    _GetThreadId.restype = DWORD
    dwThreadId = _GetThreadId(hThread)
    if dwThreadId == 0:
        raise ctypes.WinError()
    return dwThreadId