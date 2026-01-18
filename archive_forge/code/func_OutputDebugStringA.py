import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def OutputDebugStringA(lpOutputString):
    _OutputDebugStringA = windll.kernel32.OutputDebugStringA
    _OutputDebugStringA.argtypes = [LPSTR]
    _OutputDebugStringA.restype = None
    _OutputDebugStringA(lpOutputString)