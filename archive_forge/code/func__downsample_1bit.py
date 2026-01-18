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
def _downsample_1bit(pixelarray):
    byte_list = []
    value = 0
    for i, pixel in enumerate(pixelarray):
        index = i % 8
        if pixel:
            value |= 1 << index
        if index == 7:
            byte_list.append(value)
            value = 0
    return bytes(byte_list)