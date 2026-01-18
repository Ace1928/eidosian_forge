from winappdbg.win32.defines import *
from winappdbg.win32.version import bits
from winappdbg.win32.kernel32 import GetLastError, SetLastError
from winappdbg.win32.gdi32 import POINT, PPOINT, LPPOINT, RECT, PRECT, LPRECT
class GUITHREADINFO(Structure):
    _fields_ = [('cbSize', DWORD), ('flags', DWORD), ('hwndActive', HWND), ('hwndFocus', HWND), ('hwndCapture', HWND), ('hwndMenuOwner', HWND), ('hwndMoveSize', HWND), ('hwndCaret', HWND), ('rcCaret', RECT)]