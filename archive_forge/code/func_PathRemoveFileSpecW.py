from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRemoveFileSpecW(pszPath):
    _PathRemoveFileSpecW = windll.shlwapi.PathRemoveFileSpecW
    _PathRemoveFileSpecW.argtypes = [LPWSTR]
    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    _PathRemoveFileSpecW(pszPath)
    return pszPath.value