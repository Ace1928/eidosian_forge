import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def FlushViewOfFile(lpBaseAddress, dwNumberOfBytesToFlush=0):
    _FlushViewOfFile = windll.kernel32.FlushViewOfFile
    _FlushViewOfFile.argtypes = [LPVOID, SIZE_T]
    _FlushViewOfFile.restype = bool
    _FlushViewOfFile.errcheck = RaiseIfZero
    _FlushViewOfFile(lpBaseAddress, dwNumberOfBytesToFlush)