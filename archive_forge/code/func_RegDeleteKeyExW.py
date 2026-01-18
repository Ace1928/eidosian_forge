from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteKeyExW(hKeySrc, lpSubKey=None, samDesired=KEY_WOW64_32KEY):
    _RegDeleteKeyExW = windll.advapi32.RegDeleteKeyExW
    _RegDeleteKeyExW.argtypes = [HKEY, LPWSTR, REGSAM, DWORD]
    _RegDeleteKeyExW.restype = LONG
    _RegDeleteKeyExW.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteKeyExW(hKeySrc, lpSubKey, samDesired, 0)