from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsRelativeA(pszPath):
    _PathIsRelativeA = windll.shlwapi.PathIsRelativeA
    _PathIsRelativeA.argtypes = [LPSTR]
    _PathIsRelativeA.restype = bool
    return _PathIsRelativeA(pszPath)