import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetConsoleActiveScreenBuffer(hConsoleOutput=None):
    _SetConsoleActiveScreenBuffer = windll.kernel32.SetConsoleActiveScreenBuffer
    _SetConsoleActiveScreenBuffer.argytpes = [HANDLE]
    _SetConsoleActiveScreenBuffer.restype = bool
    _SetConsoleActiveScreenBuffer.errcheck = RaiseIfZero
    if hConsoleOutput is None:
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE)
    _SetConsoleActiveScreenBuffer(hConsoleOutput)