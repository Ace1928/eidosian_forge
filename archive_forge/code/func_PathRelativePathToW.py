from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathRelativePathToW(pszFrom=None, dwAttrFrom=FILE_ATTRIBUTE_DIRECTORY, pszTo=None, dwAttrTo=FILE_ATTRIBUTE_DIRECTORY):
    _PathRelativePathToW = windll.shlwapi.PathRelativePathToW
    _PathRelativePathToW.argtypes = [LPWSTR, LPWSTR, DWORD, LPWSTR, DWORD]
    _PathRelativePathToW.restype = bool
    _PathRelativePathToW.errcheck = RaiseIfZero
    if pszFrom:
        pszFrom = GetFullPathNameW(pszFrom)[0]
    else:
        pszFrom = GetCurrentDirectoryW()
    if pszTo:
        pszTo = GetFullPathNameW(pszTo)[0]
    else:
        pszTo = GetCurrentDirectoryW()
    dwPath = max((len(pszFrom) + len(pszTo)) * 2 + 1, MAX_PATH + 1)
    pszPath = ctypes.create_unicode_buffer(u'', dwPath)
    SetLastError(ERROR_INVALID_PARAMETER)
    _PathRelativePathToW(pszPath, pszFrom, dwAttrFrom, pszTo, dwAttrTo)
    return pszPath.value