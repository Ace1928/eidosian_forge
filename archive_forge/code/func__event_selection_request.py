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
@XlibEventHandler(xlib.SelectionRequest)
def _event_selection_request(self, ev):
    request = ev.xselectionrequest
    if _debug:
        rt = xlib.XGetAtomName(self._x_display, request.target)
        rp = xlib.XGetAtomName(self._x_display, request.property)
        print(f'X11 debug: request target {rt}')
        print(f'X11 debug: request property {rp}')
    out_event = xlib.XEvent()
    out_event.xany.type = xlib.SelectionNotify
    out_event.xselection.selection = request.selection
    out_event.xselection.display = request.display
    out_event.xselection.target = 0
    out_event.xselection.property = 0
    out_event.xselection.requestor = request.requestor
    out_event.xselection.time = request.time
    if xlib.XGetSelectionOwner(self._x_display, self._clipboard_atom) == self._window and ev.xselection.target == self._clipboard_atom:
        if request.target == self._target_atom:
            atoms_ar = (xlib.Atom * 1)(self._utf8_atom)
            ptr = cast(pointer(atoms_ar), POINTER(c_ubyte))
            xlib.XChangeProperty(self._x_display, request.requestor, request.property, XA_ATOM, 32, xlib.PropModeReplace, ptr, sizeof(atoms_ar) // sizeof(c_ulong))
            out_event.xselection.property = request.property
            out_event.xselection.target = request.target
        elif request.target == self._utf8_atom:
            text = self._clipboard_str.encode('utf-8')
            size = len(self._clipboard_str)
            xlib.XChangeProperty(self._x_display, request.requestor, request.property, request.target, 8, xlib.PropModeReplace, (c_ubyte * size).from_buffer_copy(text), size)
            out_event.xselection.property = request.property
            out_event.xselection.target = request.target
    xlib.XSendEvent(self._x_display, request.requestor, 0, 0, byref(out_event))