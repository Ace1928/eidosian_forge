from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathAddBackslashA(lpszPath):
    _PathAddBackslashA = windll.shlwapi.PathAddBackslashA
    _PathAddBackslashA.argtypes = [LPSTR]
    _PathAddBackslashA.restype = LPSTR
    lpszPath = ctypes.create_string_buffer(lpszPath, MAX_PATH)
    retval = _PathAddBackslashA(lpszPath)
    if retval == NULL:
        raise ctypes.WinError()
    return lpszPath.value