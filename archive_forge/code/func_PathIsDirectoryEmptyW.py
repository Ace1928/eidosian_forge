from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsDirectoryEmptyW(pszPath):
    _PathIsDirectoryEmptyW = windll.shlwapi.PathIsDirectoryEmptyW
    _PathIsDirectoryEmptyW.argtypes = [LPWSTR]
    _PathIsDirectoryEmptyW.restype = bool
    return _PathIsDirectoryEmptyW(pszPath)