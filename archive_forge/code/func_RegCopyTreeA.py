from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegCopyTreeA(hKeySrc, lpSubKey, hKeyDest):
    _RegCopyTreeA = windll.advapi32.RegCopyTreeA
    _RegCopyTreeA.argtypes = [HKEY, LPSTR, HKEY]
    _RegCopyTreeA.restype = LONG
    _RegCopyTreeA.errcheck = RaiseIfNotErrorSuccess
    _RegCopyTreeA(hKeySrc, lpSubKey, hKeyDest)