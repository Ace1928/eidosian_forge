from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathCombineW(lpszDir, lpszFile):
    _PathCombineW = windll.shlwapi.PathCombineW
    _PathCombineW.argtypes = [LPWSTR, LPWSTR, LPWSTR]
    _PathCombineW.restype = LPWSTR
    lpszDest = ctypes.create_unicode_buffer(u'', max(MAX_PATH, len(lpszDir) + len(lpszFile) + 1))
    retval = _PathCombineW(lpszDest, lpszDir, lpszFile)
    if retval == NULL:
        return None
    return lpszDest.value