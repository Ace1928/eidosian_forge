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
def _create_xdnd_atoms(self, display):
    self._xdnd_atoms = {'XdndAware': xlib.XInternAtom(display, asbytes('XdndAware'), False), 'XdndEnter': xlib.XInternAtom(display, asbytes('XdndEnter'), False), 'XdndTypeList': xlib.XInternAtom(display, asbytes('XdndTypeList'), False), 'XdndDrop': xlib.XInternAtom(display, asbytes('XdndDrop'), False), 'XdndFinished': xlib.XInternAtom(display, asbytes('XdndFinished'), False), 'XdndSelection': xlib.XInternAtom(display, asbytes('XdndSelection'), False), 'XdndPosition': xlib.XInternAtom(display, asbytes('XdndPosition'), False), 'XdndStatus': xlib.XInternAtom(display, asbytes('XdndStatus'), False), 'XdndActionCopy': xlib.XInternAtom(display, asbytes('XdndActionCopy'), False), 'text/uri-list': xlib.XInternAtom(display, asbytes('text/uri-list'), False)}