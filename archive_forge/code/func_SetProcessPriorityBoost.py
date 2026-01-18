import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetProcessPriorityBoost(hProcess, DisablePriorityBoost):
    _SetProcessPriorityBoost = windll.kernel32.SetProcessPriorityBoost
    _SetProcessPriorityBoost.argtypes = [HANDLE, BOOL]
    _SetProcessPriorityBoost.restype = bool
    _SetProcessPriorityBoost.errcheck = RaiseIfZero
    _SetProcessPriorityBoost(hProcess, bool(DisablePriorityBoost))