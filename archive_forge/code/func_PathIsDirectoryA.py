from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsDirectoryA(pszPath):
    _PathIsDirectoryA = windll.shlwapi.PathIsDirectoryA
    _PathIsDirectoryA.argtypes = [LPSTR]
    _PathIsDirectoryA.restype = bool
    return _PathIsDirectoryA(pszPath)