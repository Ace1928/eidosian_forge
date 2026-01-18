from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRemoveBackslashW(pszPath):
    _PathRemoveBackslashW = windll.shlwapi.PathRemoveBackslashW
    _PathRemoveBackslashW.argtypes = [LPWSTR]
    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathRemoveBackslashW(pszPath)
    return pszPath.value