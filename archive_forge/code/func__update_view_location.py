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
def _update_view_location(self, width, height):
    if self._fullscreen:
        x = (self.screen.width - width) // 2
        y = (self.screen.height - height) // 2
    else:
        x = y = 0
    _user32.SetWindowPos(self._view_hwnd, 0, x, y, width, height, SWP_NOZORDER | SWP_NOOWNERZORDER)