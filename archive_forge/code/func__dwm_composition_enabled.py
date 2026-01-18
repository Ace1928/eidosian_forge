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
def _dwm_composition_enabled(self):
    """ Checks if Windows DWM is enabled (Windows Vista+)
            Note: Always on for Windows 8+
        """
    is_enabled = c_int()
    _dwmapi.DwmIsCompositionEnabled(byref(is_enabled))
    return is_enabled.value