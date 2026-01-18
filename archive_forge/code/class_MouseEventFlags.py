from __future__ import annotations
import enum
import typing
from ctypes import POINTER, Structure, Union, windll
from ctypes.wintypes import BOOL, CHAR, DWORD, HANDLE, LPDWORD, SHORT, UINT, WCHAR, WORD
class MouseEventFlags(enum.IntFlag):
    """https://learn.microsoft.com/en-us/windows/console/mouse-event-record-str"""
    BUTTON_PRESSED = 0
    MOUSE_MOVED = 1
    DOUBLE_CLICK = 2
    MOUSE_WHEELED = 4
    MOUSE_HWHEELED = 8