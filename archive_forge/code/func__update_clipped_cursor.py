from ctypes import *
from functools import lru_cache
import unicodedata
from pyglet import compat_platform
import pyglet
from pyglet.window import BaseWindow, WindowException, MouseCursor
from pyglet.window import DefaultMouseCursor, _PlatformEventHandler, _ViewEventHandler
from pyglet.event import EventDispatcher
from pyglet.window import key, mouse
from pyglet.canvas.win32 import Win32Canvas
from pyglet.libs.win32 import _user32, _kernel32, _gdi32, _dwmapi, _shell32
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.winkey import *
from pyglet.libs.win32.types import *
def _update_clipped_cursor(self):
    if self._in_title_bar or self._pending_click:
        return
    rect = RECT()
    _user32.GetClientRect(self._view_hwnd, byref(rect))
    _user32.MapWindowPoints(self._view_hwnd, HWND_DESKTOP, byref(rect), 2)
    rect.top += 1
    rect.left += 1
    rect.right -= 1
    rect.bottom -= 1
    _user32.ClipCursor(byref(rect))