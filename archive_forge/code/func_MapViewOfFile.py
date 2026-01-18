import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def MapViewOfFile(hFileMappingObject, dwDesiredAccess=FILE_MAP_ALL_ACCESS | FILE_MAP_EXECUTE, dwFileOffsetHigh=0, dwFileOffsetLow=0, dwNumberOfBytesToMap=0):
    _MapViewOfFile = windll.kernel32.MapViewOfFile
    _MapViewOfFile.argtypes = [HANDLE, DWORD, DWORD, DWORD, SIZE_T]
    _MapViewOfFile.restype = LPVOID
    lpBaseAddress = _MapViewOfFile(hFileMappingObject, dwDesiredAccess, dwFileOffsetHigh, dwFileOffsetLow, dwNumberOfBytesToMap)
    if lpBaseAddress == NULL:
        raise ctypes.WinError()
    return lpBaseAddress