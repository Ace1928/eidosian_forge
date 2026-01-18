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
@ViewEventHandler
@Win32EventHandler(WM_MOUSELEAVE)
def _event_mouseleave(self, msg, wParam, lParam):
    point = POINT()
    _user32.GetCursorPos(byref(point))
    _user32.ScreenToClient(self._view_hwnd, byref(point))
    x = point.x
    y = self._height - point.y
    self._tracking = False
    self._mouse_in_window = False
    self.set_mouse_platform_visible()
    self.dispatch_event('on_mouse_leave', x, y)
    return 0