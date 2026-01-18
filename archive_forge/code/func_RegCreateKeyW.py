from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegCreateKeyW(hKey=HKEY_LOCAL_MACHINE, lpSubKey=None):
    _RegCreateKeyW = windll.advapi32.RegCreateKeyW
    _RegCreateKeyW.argtypes = [HKEY, LPWSTR, PHKEY]
    _RegCreateKeyW.restype = LONG
    _RegCreateKeyW.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegCreateKeyW(hKey, lpSubKey, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)