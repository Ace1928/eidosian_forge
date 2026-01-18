import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetCurrentDirectoryA():
    _GetCurrentDirectoryA = windll.kernel32.GetCurrentDirectoryA
    _GetCurrentDirectoryA.argtypes = [DWORD, LPSTR]
    _GetCurrentDirectoryA.restype = DWORD
    nBufferLength = _GetCurrentDirectoryA(0, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_string_buffer('', nBufferLength)
    nCopied = _GetCurrentDirectoryA(nBufferLength, lpBuffer)
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value