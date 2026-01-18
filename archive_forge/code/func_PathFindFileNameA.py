from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathFindFileNameA(pszPath):
    _PathFindFileNameA = windll.shlwapi.PathFindFileNameA
    _PathFindFileNameA.argtypes = [LPSTR]
    _PathFindFileNameA.restype = LPSTR
    pszPath = ctypes.create_string_buffer(pszPath)
    return _PathFindFileNameA(pszPath)