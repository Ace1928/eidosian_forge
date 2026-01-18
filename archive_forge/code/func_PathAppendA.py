from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathAppendA(lpszPath, pszMore=None):
    _PathAppendA = windll.shlwapi.PathAppendA
    _PathAppendA.argtypes = [LPSTR, LPSTR]
    _PathAppendA.restype = bool
    _PathAppendA.errcheck = RaiseIfZero
    if not pszMore:
        pszMore = None
    lpszPath = ctypes.create_string_buffer(lpszPath, MAX_PATH)
    _PathAppendA(lpszPath, pszMore)
    return lpszPath.value