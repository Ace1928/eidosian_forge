from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathAddExtensionW(lpszPath, pszExtension=None):
    _PathAddExtensionW = windll.shlwapi.PathAddExtensionW
    _PathAddExtensionW.argtypes = [LPWSTR, LPWSTR]
    _PathAddExtensionW.restype = bool
    _PathAddExtensionW.errcheck = RaiseIfZero
    if not pszExtension:
        pszExtension = None
    lpszPath = ctypes.create_unicode_buffer(lpszPath, MAX_PATH)
    _PathAddExtensionW(lpszPath, pszExtension)
    return lpszPath.value