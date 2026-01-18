from winappdbg.win32.defines import *
from winappdbg.win32.kernel32 import LocalFree
def ShellExecuteA(hwnd=None, lpOperation=None, lpFile=None, lpParameters=None, lpDirectory=None, nShowCmd=None):
    _ShellExecuteA = windll.shell32.ShellExecuteA
    _ShellExecuteA.argtypes = [HWND, LPSTR, LPSTR, LPSTR, LPSTR, INT]
    _ShellExecuteA.restype = HINSTANCE
    if not nShowCmd:
        nShowCmd = 0
    success = _ShellExecuteA(hwnd, lpOperation, lpFile, lpParameters, lpDirectory, nShowCmd)
    success = ctypes.cast(success, c_int)
    success = success.value
    if not success > 32:
        raise ctypes.WinError(success)