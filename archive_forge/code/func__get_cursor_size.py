import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
def _get_cursor_size(self) -> int:
    """Get the percentage of the character cell that is filled by the cursor"""
    cursor_info = CONSOLE_CURSOR_INFO()
    GetConsoleCursorInfo(self._handle, cursor_info=cursor_info)
    return int(cursor_info.dwSize)