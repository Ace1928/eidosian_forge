from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def SaferCreateLevel(dwScopeId=SAFER_SCOPEID_USER, dwLevelId=SAFER_LEVELID_NORMALUSER, OpenFlags=0):
    _SaferCreateLevel = windll.advapi32.SaferCreateLevel
    _SaferCreateLevel.argtypes = [DWORD, DWORD, DWORD, POINTER(SAFER_LEVEL_HANDLE), LPVOID]
    _SaferCreateLevel.restype = BOOL
    _SaferCreateLevel.errcheck = RaiseIfZero
    hLevelHandle = SAFER_LEVEL_HANDLE(INVALID_HANDLE_VALUE)
    _SaferCreateLevel(dwScopeId, dwLevelId, OpenFlags, byref(hLevelHandle), None)
    return SaferLevelHandle(hLevelHandle.value)