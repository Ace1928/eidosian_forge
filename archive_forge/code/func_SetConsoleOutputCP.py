import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetConsoleOutputCP(wCodePageID):
    _SetConsoleOutputCP = windll.kernel32.SetConsoleOutputCP
    _SetConsoleOutputCP.argytpes = [UINT]
    _SetConsoleOutputCP.restype = bool
    _SetConsoleOutputCP.errcheck = RaiseIfZero
    _SetConsoleOutputCP(wCodePageID)