from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsDirectoryW(pszPath):
    _PathIsDirectoryW = windll.shlwapi.PathIsDirectoryW
    _PathIsDirectoryW.argtypes = [LPWSTR]
    _PathIsDirectoryW.restype = bool
    return _PathIsDirectoryW(pszPath)