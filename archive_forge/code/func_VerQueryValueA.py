from winappdbg.win32.defines import *
def VerQueryValueA(pBlock, lpSubBlock):
    _VerQueryValueA = windll.version.VerQueryValueA
    _VerQueryValueA.argtypes = [LPVOID, LPSTR, LPVOID, POINTER(UINT)]
    _VerQueryValueA.restype = bool
    _VerQueryValueA.errcheck = RaiseIfZero
    lpBuffer = LPVOID(0)
    uLen = UINT(0)
    _VerQueryValueA(pBlock, lpSubBlock, byref(lpBuffer), byref(uLen))
    return (lpBuffer, uLen.value)