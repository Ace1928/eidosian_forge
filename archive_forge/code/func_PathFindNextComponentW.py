from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathFindNextComponentW(pszPath):
    _PathFindNextComponentW = windll.shlwapi.PathFindNextComponentW
    _PathFindNextComponentW.argtypes = [LPWSTR]
    _PathFindNextComponentW.restype = LPWSTR
    pszPath = ctypes.create_unicode_buffer(pszPath)
    return _PathFindNextComponentW(pszPath)