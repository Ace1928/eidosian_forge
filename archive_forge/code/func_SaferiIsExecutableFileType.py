from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import *
def SaferiIsExecutableFileType(szFullPath, bFromShellExecute=False):
    _SaferiIsExecutableFileType = windll.advapi32.SaferiIsExecutableFileType
    _SaferiIsExecutableFileType.argtypes = [LPWSTR, BOOLEAN]
    _SaferiIsExecutableFileType.restype = BOOL
    _SaferiIsExecutableFileType.errcheck = RaiseIfLastError
    SetLastError(ERROR_SUCCESS)
    return bool(_SaferiIsExecutableFileType(compat.unicode(szFullPath), bFromShellExecute))