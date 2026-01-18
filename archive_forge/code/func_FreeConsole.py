import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def FreeConsole():
    _FreeConsole = windll.kernel32.FreeConsole
    _FreeConsole.argytpes = []
    _FreeConsole.restype = bool
    _FreeConsole.errcheck = RaiseIfZero
    _FreeConsole()