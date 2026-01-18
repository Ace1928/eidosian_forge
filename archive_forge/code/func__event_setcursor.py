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
@Win32EventHandler(WM_SETCURSOR)
def _event_setcursor(self, msg, wParam, lParam):
    if self._exclusive_mouse and (not self._mouse_platform_visible):
        lo, hi = self._get_location(lParam)
        if lo == HTCLIENT:
            self._set_cursor_visibility(False)
            return 1
        elif lo in (HTCAPTION, HTCLOSE, HTMAXBUTTON, HTMINBUTTON):
            self._set_cursor_visibility(True)
            return 1