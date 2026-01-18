import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetEvent(hEvent):
    _SetEvent = windll.kernel32.SetEvent
    _SetEvent.argtypes = [HANDLE]
    _SetEvent.restype = bool
    _SetEvent.errcheck = RaiseIfZero
    _SetEvent(hEvent)