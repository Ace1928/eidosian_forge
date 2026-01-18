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
def _update_exclusivity(self):
    mouse_exclusive = self._active and self._mouse_exclusive
    keyboard_exclusive = self._active and self._keyboard_exclusive
    if mouse_exclusive != self._applied_mouse_exclusive:
        if mouse_exclusive:
            self.set_mouse_platform_visible(False)
            xlib.XGrabPointer(self._x_display, self._window, True, 0, xlib.GrabModeAsync, xlib.GrabModeAsync, self._window, 0, xlib.CurrentTime)
            x = self._width // 2
            y = self._height // 2
            self._mouse_exclusive_client = (x, y)
            self.set_mouse_position(x, y)
        elif self._fullscreen and (not self.screen._xinerama):
            self.set_mouse_position(0, 0)
            r = xlib.XGrabPointer(self._x_display, self._view, True, 0, xlib.GrabModeAsync, xlib.GrabModeAsync, self._view, 0, xlib.CurrentTime)
            if r:
                self._applied_mouse_exclusive = None
                return
            self.set_mouse_platform_visible()
        else:
            xlib.XUngrabPointer(self._x_display, xlib.CurrentTime)
            self.set_mouse_platform_visible()
        self._applied_mouse_exclusive = mouse_exclusive
    if keyboard_exclusive != self._applied_keyboard_exclusive:
        if keyboard_exclusive:
            xlib.XGrabKeyboard(self._x_display, self._window, False, xlib.GrabModeAsync, xlib.GrabModeAsync, xlib.CurrentTime)
        else:
            xlib.XUngrabKeyboard(self._x_display, xlib.CurrentTime)
        self._applied_keyboard_exclusive = keyboard_exclusive