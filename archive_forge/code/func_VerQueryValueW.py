from winappdbg.win32.defines import *
def VerQueryValueW(pBlock, lpSubBlock):
    _VerQueryValueW = windll.version.VerQueryValueW
    _VerQueryValueW.argtypes = [LPVOID, LPWSTR, LPVOID, POINTER(UINT)]
    _VerQueryValueW.restype = bool
    _VerQueryValueW.errcheck = RaiseIfZero
    lpBuffer = LPVOID(0)
    uLen = UINT(0)
    _VerQueryValueW(pBlock, lpSubBlock, byref(lpBuffer), byref(uLen))
    return (lpBuffer, uLen.value)