from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathFindExtensionA(pszPath):
    _PathFindExtensionA = windll.shlwapi.PathFindExtensionA
    _PathFindExtensionA.argtypes = [LPSTR]
    _PathFindExtensionA.restype = LPSTR
    pszPath = ctypes.create_string_buffer(pszPath)
    return _PathFindExtensionA(pszPath)