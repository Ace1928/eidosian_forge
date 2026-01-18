from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def IsTokenRestricted(hTokenHandle):
    _IsTokenRestricted = windll.advapi32.IsTokenRestricted
    _IsTokenRestricted.argtypes = [HANDLE]
    _IsTokenRestricted.restype = bool
    _IsTokenRestricted.errcheck = RaiseIfNotErrorSuccess
    SetLastError(ERROR_SUCCESS)
    return _IsTokenRestricted(hTokenHandle)