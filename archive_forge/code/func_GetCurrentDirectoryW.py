import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetCurrentDirectoryW():
    _GetCurrentDirectoryW = windll.kernel32.GetCurrentDirectoryW
    _GetCurrentDirectoryW.argtypes = [DWORD, LPWSTR]
    _GetCurrentDirectoryW.restype = DWORD
    nBufferLength = _GetCurrentDirectoryW(0, None)
    if nBufferLength <= 0:
        raise ctypes.WinError()
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength)
    nCopied = _GetCurrentDirectoryW(nBufferLength, lpBuffer)
    if nCopied > nBufferLength or nCopied == 0:
        raise ctypes.WinError()
    return lpBuffer.value