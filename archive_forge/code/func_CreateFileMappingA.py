import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def CreateFileMappingA(hFile, lpAttributes=None, flProtect=PAGE_EXECUTE_READWRITE, dwMaximumSizeHigh=0, dwMaximumSizeLow=0, lpName=None):
    _CreateFileMappingA = windll.kernel32.CreateFileMappingA
    _CreateFileMappingA.argtypes = [HANDLE, LPVOID, DWORD, DWORD, DWORD, LPSTR]
    _CreateFileMappingA.restype = HANDLE
    _CreateFileMappingA.errcheck = RaiseIfZero
    if lpAttributes:
        lpAttributes = ctypes.pointer(lpAttributes)
    if not lpName:
        lpName = None
    hFileMappingObject = _CreateFileMappingA(hFile, lpAttributes, flProtect, dwMaximumSizeHigh, dwMaximumSizeLow, lpName)
    return FileMappingHandle(hFileMappingObject)