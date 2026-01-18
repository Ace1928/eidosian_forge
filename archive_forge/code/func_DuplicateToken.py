from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def DuplicateToken(ExistingTokenHandle, ImpersonationLevel=SecurityImpersonation):
    _DuplicateToken = windll.advapi32.DuplicateToken
    _DuplicateToken.argtypes = [HANDLE, SECURITY_IMPERSONATION_LEVEL, PHANDLE]
    _DuplicateToken.restype = bool
    _DuplicateToken.errcheck = RaiseIfZero
    DuplicateTokenHandle = HANDLE(INVALID_HANDLE_VALUE)
    _DuplicateToken(ExistingTokenHandle, ImpersonationLevel, byref(DuplicateTokenHandle))
    return TokenHandle(DuplicateTokenHandle.value)