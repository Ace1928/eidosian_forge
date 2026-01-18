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
@XlibEventHandler(xlib.ButtonPress)
@XlibEventHandler(xlib.ButtonRelease)
def _event_button(self, ev):
    x = ev.xbutton.x
    y = self.height - ev.xbutton.y
    button = ev.xbutton.button - 1
    if button == 7 or button == 8:
        button -= 4
    modifiers = self._translate_modifiers(ev.xbutton.state)
    if ev.type == xlib.ButtonPress:
        if self._override_redirect and (not self._active):
            self.activate()
        if ev.xbutton.button == 4:
            self.dispatch_event('on_mouse_scroll', x, y, 0, 1)
        elif ev.xbutton.button == 5:
            self.dispatch_event('on_mouse_scroll', x, y, 0, -1)
        elif ev.xbutton.button == 6:
            self.dispatch_event('on_mouse_scroll', x, y, -1, 0)
        elif ev.xbutton.button == 7:
            self.dispatch_event('on_mouse_scroll', x, y, 1, 0)
        elif button < 5:
            self.dispatch_event('on_mouse_press', x, y, 1 << button, modifiers)
    elif button < 5:
        self.dispatch_event('on_mouse_release', x, y, 1 << button, modifiers)