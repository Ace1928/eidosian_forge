from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegConnectRegistryW(lpMachineName=None, hKey=HKEY_LOCAL_MACHINE):
    _RegConnectRegistryW = windll.advapi32.RegConnectRegistryW
    _RegConnectRegistryW.argtypes = [LPWSTR, HKEY, PHKEY]
    _RegConnectRegistryW.restype = LONG
    _RegConnectRegistryW.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegConnectRegistryW(lpMachineName, hKey, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)