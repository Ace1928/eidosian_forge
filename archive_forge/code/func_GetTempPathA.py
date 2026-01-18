import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetTempPathA():
    _GetTempPathA = windll.kernel32.GetTempPathA
    _GetTempPathA.argtypes = [DWORD, LPSTR]
    _GetTempPathA.restype = DWORD
    nBufferLength = _GetTempPathA(0, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_string_buffer('', nBufferLength)
    nCopied = _GetTempPathA(nBufferLength, lpBuffer)
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value