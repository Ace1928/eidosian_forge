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
def _set_atoms_property(self, name, values, mode=xlib.PropModeReplace):
    name_atom = xlib.XInternAtom(self._x_display, asbytes(name), False)
    atoms = []
    for value in values:
        atoms.append(xlib.XInternAtom(self._x_display, asbytes(value), False))
    atom_type = xlib.XInternAtom(self._x_display, asbytes('ATOM'), False)
    if len(atoms):
        atoms_ar = (xlib.Atom * len(atoms))(*atoms)
        xlib.XChangeProperty(self._x_display, self._window, name_atom, atom_type, 32, mode, cast(pointer(atoms_ar), POINTER(c_ubyte)), len(atoms))
    else:
        net_wm_state = xlib.XInternAtom(self._x_display, asbytes('_NET_WM_STATE'), False)
        if net_wm_state:
            xlib.XDeleteProperty(self._x_display, self._window, net_wm_state)