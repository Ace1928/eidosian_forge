from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathFileExistsW(pszPath):
    _PathFileExistsW = windll.shlwapi.PathFileExistsW
    _PathFileExistsW.argtypes = [LPWSTR]
    _PathFileExistsW.restype = bool
    return _PathFileExistsW(pszPath)