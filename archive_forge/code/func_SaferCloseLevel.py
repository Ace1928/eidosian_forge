from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def SaferCloseLevel(hLevelHandle):
    _SaferCloseLevel = windll.advapi32.SaferCloseLevel
    _SaferCloseLevel.argtypes = [SAFER_LEVEL_HANDLE]
    _SaferCloseLevel.restype = BOOL
    _SaferCloseLevel.errcheck = RaiseIfZero
    if hasattr(hLevelHandle, 'value'):
        _SaferCloseLevel(hLevelHandle.value)
    else:
        _SaferCloseLevel(hLevelHandle)