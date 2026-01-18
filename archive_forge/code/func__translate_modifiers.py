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
@staticmethod
def _translate_modifiers(state):
    modifiers = 0
    if state & xlib.ShiftMask:
        modifiers |= key.MOD_SHIFT
    if state & xlib.ControlMask:
        modifiers |= key.MOD_CTRL
    if state & xlib.LockMask:
        modifiers |= key.MOD_CAPSLOCK
    if state & xlib.Mod1Mask:
        modifiers |= key.MOD_ALT
    if state & xlib.Mod2Mask:
        modifiers |= key.MOD_NUMLOCK
    if state & xlib.Mod4Mask:
        modifiers |= key.MOD_WINDOWS
    if state & xlib.Mod5Mask:
        modifiers |= key.MOD_SCROLLLOCK
    return modifiers