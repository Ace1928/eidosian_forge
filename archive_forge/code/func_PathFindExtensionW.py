from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathFindExtensionW(pszPath):
    _PathFindExtensionW = windll.shlwapi.PathFindExtensionW
    _PathFindExtensionW.argtypes = [LPWSTR]
    _PathFindExtensionW.restype = LPWSTR
    pszPath = ctypes.create_unicode_buffer(pszPath)
    return _PathFindExtensionW(pszPath)