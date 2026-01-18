from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathGetArgsW(pszPath):
    _PathGetArgsW = windll.shlwapi.PathGetArgsW
    _PathGetArgsW.argtypes = [LPWSTR]
    _PathGetArgsW.restype = LPWSTR
    pszPath = ctypes.create_unicode_buffer(pszPath)
    return _PathGetArgsW(pszPath)