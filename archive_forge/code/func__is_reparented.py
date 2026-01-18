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
def _is_reparented(self):
    root = c_ulong()
    parent = c_ulong()
    children = pointer(c_ulong())
    n_children = c_uint()
    xlib.XQueryTree(self._x_display, self._window, byref(root), byref(parent), byref(children), byref(n_children))
    return root.value != parent.value