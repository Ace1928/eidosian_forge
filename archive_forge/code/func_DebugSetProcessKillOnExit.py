import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def DebugSetProcessKillOnExit(KillOnExit):
    _DebugSetProcessKillOnExit = windll.kernel32.DebugSetProcessKillOnExit
    _DebugSetProcessKillOnExit.argtypes = [BOOL]
    _DebugSetProcessKillOnExit.restype = bool
    _DebugSetProcessKillOnExit.errcheck = RaiseIfZero
    _DebugSetProcessKillOnExit(bool(KillOnExit))