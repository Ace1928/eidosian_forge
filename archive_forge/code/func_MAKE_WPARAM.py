from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
def MAKE_WPARAM(wParam):
    """
    Convert arguments to the WPARAM type.
    Used automatically by SendMessage, PostMessage, etc.
    You shouldn't need to call this function.
    """
    wParam = ctypes.cast(wParam, LPVOID).value
    if wParam is None:
        wParam = 0
    return wParam