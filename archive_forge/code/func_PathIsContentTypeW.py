from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsContentTypeW(pszPath, pszContentType):
    _PathIsContentTypeW = windll.shlwapi.PathIsContentTypeW
    _PathIsContentTypeW.argtypes = [LPWSTR, LPWSTR]
    _PathIsContentTypeW.restype = bool
    return _PathIsContentTypeW(pszPath, pszContentType)