import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetConsoleOutputCP():
    _GetConsoleOutputCP = windll.kernel32.GetConsoleOutputCP
    _GetConsoleOutputCP.argytpes = []
    _GetConsoleOutputCP.restype = UINT
    return _GetConsoleOutputCP()