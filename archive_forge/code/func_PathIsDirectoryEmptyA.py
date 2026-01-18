from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsDirectoryEmptyA(pszPath):
    _PathIsDirectoryEmptyA = windll.shlwapi.PathIsDirectoryEmptyA
    _PathIsDirectoryEmptyA.argtypes = [LPSTR]
    _PathIsDirectoryEmptyA.restype = bool
    return _PathIsDirectoryEmptyA(pszPath)