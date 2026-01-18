import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetDllDirectoryW():
    _GetDllDirectoryW = windll.kernel32.GetDllDirectoryW
    _GetDllDirectoryW.argytpes = [DWORD, LPWSTR]
    _GetDllDirectoryW.restype = DWORD
    nBufferLength = _GetDllDirectoryW(0, None)
    if nBufferLength == 0:
        return None
    lpBuffer = ctypes.create_unicode_buffer(u'', nBufferLength)
    _GetDllDirectoryW(nBufferLength, byref(lpBuffer))
    return lpBuffer.value