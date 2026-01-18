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
@XlibEventHandler(xlib.MotionNotify)
def _event_motionnotify_view(self, ev):
    x = ev.xmotion.x
    y = self.height - ev.xmotion.y - 1
    if self._mouse_in_window:
        dx = x - self._mouse_x
        dy = y - self._mouse_y
    else:
        dx = dy = 0
    if self._applied_mouse_exclusive and (ev.xmotion.x, ev.xmotion.y) == self._mouse_exclusive_client:
        self._mouse_x = x
        self._mouse_y = y
        return
    if self._applied_mouse_exclusive:
        ex, ey = self._mouse_exclusive_client
        xlib.XWarpPointer(self._x_display, 0, self._window, 0, 0, 0, 0, ex, ey)
    self._mouse_x = x
    self._mouse_y = y
    self._mouse_in_window = True
    buttons = 0
    if ev.xmotion.state & xlib.Button1MotionMask:
        buttons |= mouse.LEFT
    if ev.xmotion.state & xlib.Button2MotionMask:
        buttons |= mouse.MIDDLE
    if ev.xmotion.state & xlib.Button3MotionMask:
        buttons |= mouse.RIGHT
    if buttons:
        modifiers = self._translate_modifiers(ev.xmotion.state)
        self.dispatch_event('on_mouse_drag', x, y, dx, dy, buttons, modifiers)
    else:
        self.dispatch_event('on_mouse_motion', x, y, dx, dy)