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
@XlibEventHandler(xlib.KeyPress)
@XlibEventHandler(xlib.KeyRelease)
def _event_key_view(self, ev):
    global _can_detect_autorepeat
    if not _can_detect_autorepeat and ev.type == xlib.KeyRelease:
        saved = []
        while True:
            auto_event = xlib.XEvent()
            result = xlib.XCheckWindowEvent(self._x_display, self._window, xlib.KeyPress | xlib.KeyRelease, byref(auto_event))
            if not result:
                break
            saved.append(auto_event)
            if auto_event.type == xlib.KeyRelease:
                continue
            if ev.xkey.keycode == auto_event.xkey.keycode:
                text, symbol = self._event_text_symbol(auto_event)
                modifiers = self._translate_modifiers(ev.xkey.state)
                modifiers_ctrl = modifiers & (key.MOD_CTRL | key.MOD_ALT)
                motion = self._event_text_motion(symbol, modifiers)
                if motion:
                    if modifiers & key.MOD_SHIFT:
                        self.dispatch_event('on_text_motion_select', motion)
                    else:
                        self.dispatch_event('on_text_motion', motion)
                elif text and (not modifiers_ctrl):
                    self.dispatch_event('on_text', text)
                ditched = saved.pop()
                for auto_event in reversed(saved):
                    xlib.XPutBackEvent(self._x_display, byref(auto_event))
                return
            else:
                break
        for auto_event in reversed(saved):
            xlib.XPutBackEvent(self._x_display, byref(auto_event))
    text, symbol = self._event_text_symbol(ev)
    modifiers = self._translate_modifiers(ev.xkey.state)
    modifiers_ctrl = modifiers & (key.MOD_CTRL | key.MOD_ALT)
    motion = self._event_text_motion(symbol, modifiers)
    if ev.type == xlib.KeyPress:
        if symbol and (not _can_detect_autorepeat or symbol not in self.pressed_keys):
            self.dispatch_event('on_key_press', symbol, modifiers)
            if _can_detect_autorepeat:
                self.pressed_keys.add(symbol)
        if motion:
            if modifiers & key.MOD_SHIFT:
                self.dispatch_event('on_text_motion_select', motion)
            else:
                self.dispatch_event('on_text_motion', motion)
        elif text and (not modifiers_ctrl):
            self.dispatch_event('on_text', text)
    elif ev.type == xlib.KeyRelease:
        if symbol:
            self.dispatch_event('on_key_release', symbol, modifiers)
            if _can_detect_autorepeat and symbol in self.pressed_keys:
                self.pressed_keys.remove(symbol)