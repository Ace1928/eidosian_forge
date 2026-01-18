from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def GetUserNameA():
    _GetUserNameA = windll.advapi32.GetUserNameA
    _GetUserNameA.argtypes = [LPSTR, LPDWORD]
    _GetUserNameA.restype = bool
    nSize = DWORD(0)
    _GetUserNameA(None, byref(nSize))
    error = GetLastError()
    if error != ERROR_INSUFFICIENT_BUFFER:
        raise ctypes.WinError(error)
    lpBuffer = ctypes.create_string_buffer('', nSize.value + 1)
    success = _GetUserNameA(lpBuffer, byref(nSize))
    if not success:
        raise ctypes.WinError()
    return lpBuffer.value