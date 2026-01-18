from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsRootA(pszPath):
    _PathIsRootA = windll.shlwapi.PathIsRootA
    _PathIsRootA.argtypes = [LPSTR]
    _PathIsRootA.restype = bool
    return _PathIsRootA(pszPath)