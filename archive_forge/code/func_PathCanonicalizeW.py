from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def PathCanonicalizeW(lpszSrc):
    _PathCanonicalizeW = windll.shlwapi.PathCanonicalizeW
    _PathCanonicalizeW.argtypes = [LPWSTR, LPWSTR]
    _PathCanonicalizeW.restype = bool
    _PathCanonicalizeW.errcheck = RaiseIfZero
    lpszDst = ctypes.create_unicode_buffer(u'', MAX_PATH)
    _PathCanonicalizeW(lpszDst, lpszSrc)
    return lpszDst.value