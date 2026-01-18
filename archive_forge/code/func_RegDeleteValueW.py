from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteValueW(hKeySrc, lpValueName=None):
    _RegDeleteValueW = windll.advapi32.RegDeleteValueW
    _RegDeleteValueW.argtypes = [HKEY, LPWSTR]
    _RegDeleteValueW.restype = LONG
    _RegDeleteValueW.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteValueW(hKeySrc, lpValueName)