from __future__ import annotations
import enum
import typing
from ctypes import POINTER, Structure, Union, windll
from ctypes.wintypes import BOOL, CHAR, DWORD, HANDLE, LPDWORD, SHORT, UINT, WCHAR, WORD
class MouseButtonState(enum.IntFlag):
    """https://learn.microsoft.com/en-us/windows/console/mouse-event-record-str"""
    FROM_LEFT_1ST_BUTTON_PRESSED = 1
    RIGHTMOST_BUTTON_PRESSED = 2
    FROM_LEFT_2ND_BUTTON_PRESSED = 4
    FROM_LEFT_3RD_BUTTON_PRESSED = 8
    FROM_LEFT_4TH_BUTTON_PRESSED = 16