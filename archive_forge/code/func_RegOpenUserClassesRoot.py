from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegOpenUserClassesRoot(hToken, samDesired=KEY_ALL_ACCESS):
    _RegOpenUserClassesRoot = windll.advapi32.RegOpenUserClassesRoot
    _RegOpenUserClassesRoot.argtypes = [HANDLE, DWORD, REGSAM, PHKEY]
    _RegOpenUserClassesRoot.restype = LONG
    _RegOpenUserClassesRoot.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegOpenUserClassesRoot(hToken, 0, samDesired, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)