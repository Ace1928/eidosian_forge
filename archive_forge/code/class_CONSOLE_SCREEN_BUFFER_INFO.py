from ctypes import Union, Structure, c_char, c_short, c_long, c_ulong
from ctypes.wintypes import DWORD, BOOL, LPVOID, WORD, WCHAR
class CONSOLE_SCREEN_BUFFER_INFO(Structure):
    """struct in wincon.h."""
    _fields_ = [('dwSize', COORD), ('dwCursorPosition', COORD), ('wAttributes', WORD), ('srWindow', SMALL_RECT), ('dwMaximumWindowSize', COORD)]

    def __str__(self):
        return '(%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d)' % (self.dwSize.Y, self.dwSize.X, self.dwCursorPosition.Y, self.dwCursorPosition.X, self.wAttributes, self.srWindow.Top, self.srWindow.Left, self.srWindow.Bottom, self.srWindow.Right, self.dwMaximumWindowSize.Y, self.dwMaximumWindowSize.X)