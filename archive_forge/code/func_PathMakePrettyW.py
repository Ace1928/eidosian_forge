from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathMakePrettyW(pszPath):
    _PathMakePrettyW = windll.shlwapi.PathMakePrettyW
    _PathMakePrettyW.argtypes = [LPWSTR]
    _PathMakePrettyW.restype = bool
    _PathMakePrettyW.errcheck = RaiseIfZero
    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathMakePrettyW(pszPath)
    return pszPath.value