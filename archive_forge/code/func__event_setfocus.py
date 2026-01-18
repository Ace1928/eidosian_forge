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
@Win32EventHandler(WM_SETFOCUS)
def _event_setfocus(self, msg, wParam, lParam):
    self.dispatch_event('on_activate')
    self._has_focus = True
    if self._exclusive_mouse:
        if _user32.GetAsyncKeyState(VK_LBUTTON):
            self._pending_click = True
    self.set_exclusive_keyboard(self._exclusive_keyboard)
    self.set_exclusive_mouse(self._exclusive_mouse)
    return 0