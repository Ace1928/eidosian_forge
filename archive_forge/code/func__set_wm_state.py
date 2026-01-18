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
def _set_wm_state(self, *states):
    net_wm_state = xlib.XInternAtom(self._x_display, asbytes('_NET_WM_STATE'), False)
    atoms = []
    for state in states:
        atoms.append(xlib.XInternAtom(self._x_display, asbytes(state), False))
    atom_type = xlib.XInternAtom(self._x_display, asbytes('ATOM'), False)
    if len(atoms):
        atoms_ar = (xlib.Atom * len(atoms))(*atoms)
        xlib.XChangeProperty(self._x_display, self._window, net_wm_state, atom_type, 32, xlib.PropModePrepend, cast(pointer(atoms_ar), POINTER(c_ubyte)), len(atoms))
    else:
        xlib.XDeleteProperty(self._x_display, self._window, net_wm_state)
    e = xlib.XEvent()
    e.xclient.type = xlib.ClientMessage
    e.xclient.message_type = net_wm_state
    e.xclient.display = cast(self._x_display, POINTER(xlib.Display))
    e.xclient.window = self._window
    e.xclient.format = 32
    e.xclient.data.l[0] = xlib.PropModePrepend
    for i, atom in enumerate(atoms):
        e.xclient.data.l[i + 1] = atom
    xlib.XSendEvent(self._x_display, self._get_root(), False, xlib.SubstructureRedirectMask, byref(e))