import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetConsoleCP(wCodePageID):
    _SetConsoleCP = windll.kernel32.SetConsoleCP
    _SetConsoleCP.argytpes = [UINT]
    _SetConsoleCP.restype = bool
    _SetConsoleCP.errcheck = RaiseIfZero
    _SetConsoleCP(wCodePageID)