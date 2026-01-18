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
@Win32EventHandler(WM_CAPTURECHANGED)
def _event_capture_changed(self, msg, wParam, lParam):
    self._in_title_bar = False
    if self._exclusive_mouse:
        state = _user32.GetAsyncKeyState(VK_LBUTTON)
        if not state & 32768:
            if self._pending_click:
                self._pending_click = False
            if self._has_focus or not self._hidden:
                self._update_clipped_cursor()