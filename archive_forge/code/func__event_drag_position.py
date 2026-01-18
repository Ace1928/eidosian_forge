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
def _event_drag_position(self, ev):
    if self._xdnd_version > XDND_VERSION:
        return
    xoff = ev.xclient.data.l[2] >> 16 & 65535
    yoff = ev.xclient.data.l[2] & 65535
    child = xlib.Window()
    x = c_int()
    y = c_int()
    xlib.XTranslateCoordinates(self._x_display, self._get_root(), self._window, xoff, yoff, byref(x), byref(y), byref(child))
    self._xdnd_position = (x.value, y.value)
    e = xlib.XEvent()
    e.xclient.type = xlib.ClientMessage
    e.xclient.message_type = self._xdnd_atoms['XdndStatus']
    e.xclient.display = cast(self._x_display, POINTER(xlib.Display))
    e.xclient.window = ev.xclient.data.l[0]
    e.xclient.format = 32
    e.xclient.data.l[0] = self._window
    e.xclient.data.l[2] = 0
    e.xclient.data.l[3] = 0
    if self._xdnd_format:
        e.xclient.data.l[1] = 1
        if self._xdnd_version >= 2:
            e.xclient.data.l[4] = self._xdnd_atoms['XdndActionCopy']
    xlib.XSendEvent(self._x_display, self._xdnd_source, False, xlib.NoEventMask, byref(e))
    xlib.XFlush(self._x_display)