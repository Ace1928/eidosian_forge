import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def SetConsoleTextAttribute(hConsoleOutput=None, wAttributes=0):
    _SetConsoleTextAttribute = windll.kernel32.SetConsoleTextAttribute
    _SetConsoleTextAttribute.argytpes = [HANDLE, WORD]
    _SetConsoleTextAttribute.restype = bool
    _SetConsoleTextAttribute.errcheck = RaiseIfZero
    if hConsoleOutput is None:
        hConsoleOutput = GetStdHandle(STD_OUTPUT_HANDLE)
    _SetConsoleTextAttribute(hConsoleOutput, wAttributes)