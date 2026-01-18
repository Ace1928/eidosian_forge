from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegDeleteTreeW(hKey, lpSubKey=None):
    _RegDeleteTreeW = windll.advapi32.RegDeleteTreeW
    _RegDeleteTreeW.argtypes = [HKEY, LPWSTR]
    _RegDeleteTreeW.restype = LONG
    _RegDeleteTreeW.errcheck = RaiseIfNotErrorSuccess
    _RegDeleteTreeW(hKey, lpSubKey)