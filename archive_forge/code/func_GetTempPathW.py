import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetTempPathW():
    _GetTempPathW = windll.kernel32.GetTempPathW
    _GetTempPathW.argtypes = [DWORD, LPWSTR]
    _GetTempPathW.restype = DWORD
    nBufferLength = _GetTempPathW(0, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength)
    nCopied = _GetTempPathW(nBufferLength, lpBuffer)
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value