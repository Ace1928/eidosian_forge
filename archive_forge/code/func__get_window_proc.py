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
def _get_window_proc(self, event_handlers):

    def f(hwnd, msg, wParam, lParam):
        event_handler = event_handlers.get(msg, None)
        result = None
        if event_handler:
            if self._allow_dispatch_event or not self._enable_event_queue:
                result = event_handler(msg, wParam, lParam)
            else:
                result = 0
                self._event_queue.append((event_handler, msg, wParam, lParam))
        if result is None:
            result = _user32.DefWindowProcW(hwnd, msg, wParam, lParam)
        return result
    return f