from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathMakePrettyA(pszPath):
    _PathMakePrettyA = windll.shlwapi.PathMakePrettyA
    _PathMakePrettyA.argtypes = [LPSTR]
    _PathMakePrettyA.restype = bool
    _PathMakePrettyA.errcheck = RaiseIfZero
    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathMakePrettyA(pszPath)
    return pszPath.value