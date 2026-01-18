from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRemoveBackslashA(pszPath):
    _PathRemoveBackslashA = windll.shlwapi.PathRemoveBackslashA
    _PathRemoveBackslashA.argtypes = [LPSTR]
    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathRemoveBackslashA(pszPath)
    return pszPath.value