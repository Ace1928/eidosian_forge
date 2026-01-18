from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathFindFileNameW(pszPath):
    _PathFindFileNameW = windll.shlwapi.PathFindFileNameW
    _PathFindFileNameW.argtypes = [LPWSTR]
    _PathFindFileNameW.restype = LPWSTR
    pszPath = ctypes.create_unicode_buffer(pszPath)
    return _PathFindFileNameW(pszPath)