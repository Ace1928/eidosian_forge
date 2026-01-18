from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsRelativeW(pszPath):
    _PathIsRelativeW = windll.shlwapi.PathIsRelativeW
    _PathIsRelativeW.argtypes = [LPWSTR]
    _PathIsRelativeW.restype = bool
    return _PathIsRelativeW(pszPath)