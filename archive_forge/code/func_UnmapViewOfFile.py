import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def UnmapViewOfFile(lpBaseAddress):
    _UnmapViewOfFile = windll.kernel32.UnmapViewOfFile
    _UnmapViewOfFile.argtypes = [LPVOID]
    _UnmapViewOfFile.restype = bool
    _UnmapViewOfFile.errcheck = RaiseIfZero
    _UnmapViewOfFile(lpBaseAddress)