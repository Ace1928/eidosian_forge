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
@ViewEventHandler
@XlibEventHandler(xlib.LeaveNotify)
def _event_leavenotify(self, ev):
    x = self._mouse_x = ev.xcrossing.x
    y = self._mouse_y = self.height - ev.xcrossing.y
    self._mouse_in_window = False
    self.dispatch_event('on_mouse_leave', x, y)