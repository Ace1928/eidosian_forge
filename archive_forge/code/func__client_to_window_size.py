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
def _client_to_window_size(self, width, height):
    rect = RECT()
    rect.left = 0
    rect.top = 0
    rect.right = width
    rect.bottom = height
    _user32.AdjustWindowRectEx(byref(rect), self._ws_style, False, self._ex_ws_style)
    return (rect.right - rect.left, rect.bottom - rect.top)