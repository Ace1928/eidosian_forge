from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def RegOpenCurrentUser(samDesired=KEY_ALL_ACCESS):
    _RegOpenCurrentUser = windll.advapi32.RegOpenCurrentUser
    _RegOpenCurrentUser.argtypes = [REGSAM, PHKEY]
    _RegOpenCurrentUser.restype = LONG
    _RegOpenCurrentUser.errcheck = RaiseIfNotErrorSuccess
    hkResult = HKEY(INVALID_HANDLE_VALUE)
    _RegOpenCurrentUser(samDesired, byref(hkResult))
    return RegistryKeyHandle(hkResult.value)