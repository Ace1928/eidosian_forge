from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathIsContentTypeA(pszPath, pszContentType):
    _PathIsContentTypeA = windll.shlwapi.PathIsContentTypeA
    _PathIsContentTypeA.argtypes = [LPSTR, LPSTR]
    _PathIsContentTypeA.restype = bool
    return _PathIsContentTypeA(pszPath, pszContentType)