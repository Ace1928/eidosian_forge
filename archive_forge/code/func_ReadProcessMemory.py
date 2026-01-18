import warnings
from winappdbg.win32.defines import *
from winappdbg.win32 import context_i386
from winappdbg.win32 import context_amd64
from winappdbg.win32.version import *
def ReadProcessMemory(hProcess, lpBaseAddress, nSize):
    _ReadProcessMemory = windll.kernel32.ReadProcessMemory
    _ReadProcessMemory.argtypes = [HANDLE, LPVOID, LPVOID, SIZE_T, POINTER(SIZE_T)]
    _ReadProcessMemory.restype = bool
    lpBuffer = ctypes.create_string_buffer(compat.b(''), nSize)
    lpNumberOfBytesRead = SIZE_T(0)
    success = _ReadProcessMemory(hProcess, lpBaseAddress, lpBuffer, nSize, byref(lpNumberOfBytesRead))
    if not success and GetLastError() != ERROR_PARTIAL_COPY:
        raise ctypes.WinError()
    return compat.b(lpBuffer.raw)[:lpNumberOfBytesRead.value]