from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsNetworkPathA(pszPath):
    _PathIsNetworkPathA = windll.shlwapi.PathIsNetworkPathA
    _PathIsNetworkPathA.argtypes = [LPSTR]
    _PathIsNetworkPathA.restype = bool
    return _PathIsNetworkPathA(pszPath)