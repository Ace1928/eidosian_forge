import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def ResetEvent(hEvent):
    _ResetEvent = windll.kernel32.ResetEvent
    _ResetEvent.argtypes = [HANDLE]
    _ResetEvent.restype = bool
    _ResetEvent.errcheck = RaiseIfZero
    _ResetEvent(hEvent)