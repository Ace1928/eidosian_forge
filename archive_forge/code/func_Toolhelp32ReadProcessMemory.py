import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def Toolhelp32ReadProcessMemory(th32ProcessID, lpBaseAddress, cbRead):
    _Toolhelp32ReadProcessMemory = windll.kernel32.Toolhelp32ReadProcessMemory
    _Toolhelp32ReadProcessMemory.argtypes = [DWORD, LPVOID, LPVOID, SIZE_T, POINTER(SIZE_T)]
    _Toolhelp32ReadProcessMemory.restype = bool
    lpBuffer = ctypes.create_string_buffer('', cbRead)
    lpNumberOfBytesRead = SIZE_T(0)
    success = _Toolhelp32ReadProcessMemory(th32ProcessID, lpBaseAddress, lpBuffer, cbRead, byref(lpNumberOfBytesRead))
    if not success and GetLastError() != ERROR_PARTIAL_COPY:
        raise ctypes.WinError()
    return str(lpBuffer.raw)[:lpNumberOfBytesRead.value]