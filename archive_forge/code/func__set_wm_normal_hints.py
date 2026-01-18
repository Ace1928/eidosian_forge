import locale
import unicodedata
import urllib.parse
from ctypes import *
from functools import lru_cache
from typing import Optional
import pyglet
from pyglet.window import WindowException, MouseCursorException
from pyglet.window import MouseCursor, DefaultMouseCursor, ImageMouseCursor
from pyglet.window import BaseWindow, _PlatformEventHandler, _ViewEventHandler
from pyglet.window import key
from pyglet.window import mouse
from pyglet.event import EventDispatcher
from pyglet.canvas.xlib import XlibCanvas
from pyglet.libs.x11 import xlib
from pyglet.libs.x11 import cursorfont
from pyglet.util import asbytes
def _set_wm_normal_hints(self):
    hints = xlib.XAllocSizeHints().contents
    if self._minimum_size:
        hints.flags |= xlib.PMinSize
        hints.min_width, hints.min_height = self._minimum_size
    if self._maximum_size:
        hints.flags |= xlib.PMaxSize
        hints.max_width, hints.max_height = self._maximum_size
    xlib.XSetWMNormalHints(self._x_display, self._window, byref(hints))