from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRenameExtensionA(pszPath, pszExt):
    _PathRenameExtensionA = windll.shlwapi.PathRenameExtensionA
    _PathRenameExtensionA.argtypes = [LPSTR, LPSTR]
    _PathRenameExtensionA.restype = bool
    pszPath = ctypes.create_string_buffer(pszPath, MAX_PATH)
    if _PathRenameExtensionA(pszPath, pszExt):
        return pszPath.value
    return None