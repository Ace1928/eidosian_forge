from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRemoveFileSpecA(pszPath):
    _PathRemoveFileSpecA = windll.shlwapi.PathRemoveFileSpecA
    _PathRemoveFileSpecA.argtypes = [LPSTR]
    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    _PathRemoveFileSpecA(pszPath)
    return pszPath.value