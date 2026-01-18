from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegOpenKeyA(hKey=HKEY_LOCAL_MACHINE, lpSubKey=None):
    _RegOpenKeyA = windll.advapi32.RegOpenKeyA
    _RegOpenKeyA.argtypes = [HKEY, LPSTR, PHKEY]
    _RegOpenKeyA.restype = LONG
    _RegOpenKeyA.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegOpenKeyA(hKey, lpSubKey, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)