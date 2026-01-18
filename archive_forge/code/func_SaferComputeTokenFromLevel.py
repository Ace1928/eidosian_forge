from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def SaferComputeTokenFromLevel(LevelHandle, InAccessToken=None, dwFlags=0):
    _SaferComputeTokenFromLevel = windll.advapi32.SaferComputeTokenFromLevel
    _SaferComputeTokenFromLevel.argtypes = [SAFER_LEVEL_HANDLE, HANDLE, PHANDLE, DWORD, LPDWORD]
    _SaferComputeTokenFromLevel.restype = BOOL
    _SaferComputeTokenFromLevel.errcheck = RaiseIfZero
    OutAccessToken = HANDLE(INVALID_HANDLE_VALUE)
    lpReserved = DWORD(0)
    _SaferComputeTokenFromLevel(LevelHandle, InAccessToken, byref(OutAccessToken), dwFlags, byref(lpReserved))
    return (TokenHandle(OutAccessToken.value), lpReserved.value)