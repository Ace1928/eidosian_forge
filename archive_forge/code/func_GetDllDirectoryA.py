import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def GetDllDirectoryA():
    _GetDllDirectoryA = windll.kernel32.GetDllDirectoryA
    _GetDllDirectoryA.argytpes = [DWORD, LPSTR]
    _GetDllDirectoryA.restype = DWORD
    nBufferLength = _GetDllDirectoryA(0, None)
    if nBufferLength == 0:
        return None
    lpBuffer = ctypes.create_string_buffer('', nBufferLength)
    _GetDllDirectoryA(nBufferLength, byref(lpBuffer))
    return lpBuffer.value