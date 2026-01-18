from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsUNCW(pszPath):
    _PathIsUNCW = windll.shlwapi.PathIsUNCW
    _PathIsUNCW.argtypes = [LPWSTR]
    _PathIsUNCW.restype = bool
    return _PathIsUNCW(pszPath)