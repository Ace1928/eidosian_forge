from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def DuplicateTokenEx(hExistingToken, dwDesiredAccess=TOKEN_ALL_ACCESS, lpTokenAttributes=None, ImpersonationLevel=SecurityImpersonation, TokenType=TokenPrimary):
    _DuplicateTokenEx = windll.advapi32.DuplicateTokenEx
    _DuplicateTokenEx.argtypes = [HANDLE, DWORD, LPSECURITY_ATTRIBUTES, SECURITY_IMPERSONATION_LEVEL, TOKEN_TYPE, PHANDLE]
    _DuplicateTokenEx.restype = bool
    _DuplicateTokenEx.errcheck = RaiseIfZero
    DuplicateTokenHandle = HANDLE(INVALID_HANDLE_VALUE)
    _DuplicateTokenEx(hExistingToken, dwDesiredAccess, lpTokenAttributes, ImpersonationLevel, TokenType, byref(DuplicateTokenHandle))
    return TokenHandle(DuplicateTokenHandle.value)