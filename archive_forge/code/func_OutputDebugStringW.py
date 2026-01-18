import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def OutputDebugStringW(lpOutputString):
    _OutputDebugStringW = windll.kernel32.OutputDebugStringW
    _OutputDebugStringW.argtypes = [LPWSTR]
    _OutputDebugStringW.restype = None
    _OutputDebugStringW(lpOutputString)