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
def _set_text_property(self, name, value, allow_utf8=True):
    atom = xlib.XInternAtom(self._x_display, asbytes(name), False)
    if not atom:
        raise XlibException('Undefined atom "%s"' % name)
    text_property = xlib.XTextProperty()
    if _have_utf8 and allow_utf8:
        buf = create_string_buffer(value.encode('utf8'))
        result = xlib.Xutf8TextListToTextProperty(self._x_display, cast(pointer(buf), c_char_p), 1, xlib.XUTF8StringStyle, byref(text_property))
        if result < 0:
            raise XlibException('Could not create UTF8 text property')
    else:
        buf = create_string_buffer(value.encode('ascii', 'ignore'))
        result = xlib.XStringListToTextProperty(cast(pointer(buf), c_char_p), 1, byref(text_property))
        if result < 0:
            raise XlibException('Could not create text property')
    xlib.XSetTextProperty(self._x_display, self._window, byref(text_property), atom)