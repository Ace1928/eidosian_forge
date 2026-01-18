from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def FindExecutableW(lpFile, lpDirectory=None):
    _FindExecutableW = windll.shell32.FindExecutableW
    _FindExecutableW.argtypes = [LPWSTR, LPWSTR, LPWSTR]
    _FindExecutableW.restype = HINSTANCE
    lpResult = ctypes.create_unicode_buffer(MAX_PATH)
    success = _FindExecutableW(lpFile, lpDirectory, lpResult)
    success = ctypes.cast(success, ctypes.c_void_p)
    success = success.value
    if not success > 32:
        raise ctypes.WinError(success)
    return lpResult.value