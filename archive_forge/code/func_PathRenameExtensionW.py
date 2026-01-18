from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRenameExtensionW(pszPath, pszExt):
    _PathRenameExtensionW = windll.shlwapi.PathRenameExtensionW
    _PathRenameExtensionW.argtypes = [LPWSTR, LPWSTR]
    _PathRenameExtensionW.restype = bool
    pszPath = ctypes.create_unicode_buffer(pszPath, MAX_PATH)
    if _PathRenameExtensionW(pszPath, pszExt):
        return pszPath.value
    return None