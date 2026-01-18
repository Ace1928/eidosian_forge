from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathGetArgsA(pszPath):
    _PathGetArgsA = windll.shlwapi.PathGetArgsA
    _PathGetArgsA.argtypes = [LPSTR]
    _PathGetArgsA.restype = LPSTR
    pszPath = ctypes.create_string_buffer(pszPath)
    return _PathGetArgsA(pszPath)