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
def _event_drag_enter(self, ev):
    self._xdnd_source = ev.xclient.data.l[0]
    self._xdnd_version = ev.xclient.data.l[1] >> 24
    self._xdnd_format = None
    if self._xdnd_version > XDND_VERSION:
        return
    three_or_more = ev.xclient.data.l[1] & 1
    if three_or_more:
        data, count, _ = self.get_single_property(self._xdnd_source, self._xdnd_atoms['XdndTypeList'], XA_ATOM)
        data = cast(data, POINTER(xlib.Atom))
    else:
        count = 3
        data = ev.xclient.data.l + 2
    for i in range(count):
        if data[i] == self._xdnd_atoms['text/uri-list']:
            self._xdnd_format = self._xdnd_atoms['text/uri-list']
            break
    if data:
        xlib.XFree(data)