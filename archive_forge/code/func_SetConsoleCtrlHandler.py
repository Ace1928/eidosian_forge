import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetConsoleCtrlHandler(HandlerRoutine=None, Add=True):
    _SetConsoleCtrlHandler = windll.kernel32.SetConsoleCtrlHandler
    _SetConsoleCtrlHandler.argtypes = [PHANDLER_ROUTINE, BOOL]
    _SetConsoleCtrlHandler.restype = bool
    _SetConsoleCtrlHandler.errcheck = RaiseIfZero
    _SetConsoleCtrlHandler(HandlerRoutine, bool(Add))