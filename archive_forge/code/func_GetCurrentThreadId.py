import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetCurrentThreadId():
    _GetCurrentThreadId = windll.kernel32.GetCurrentThreadId
    _GetCurrentThreadId.argtypes = []
    _GetCurrentThreadId.restype = DWORD
    return _GetCurrentThreadId()