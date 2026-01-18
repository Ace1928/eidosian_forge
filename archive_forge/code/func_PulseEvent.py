import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def PulseEvent(hEvent):
    _PulseEvent = windll.kernel32.PulseEvent
    _PulseEvent.argtypes = [HANDLE]
    _PulseEvent.restype = bool
    _PulseEvent.errcheck = RaiseIfZero
    _PulseEvent(hEvent)