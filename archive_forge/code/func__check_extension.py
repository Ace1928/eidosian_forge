import ctypes
import pyglet
from pyglet.input.base import Device, DeviceOpenException
from pyglet.input.base import Button, RelativeAxis, AbsoluteAxis
from pyglet.libs.x11 import xlib
from pyglet.util import asstr
def _check_extension(display):
    major_opcode = ctypes.c_int()
    first_event = ctypes.c_int()
    first_error = ctypes.c_int()
    xlib.XQueryExtension(display._display, b'XInputExtension', ctypes.byref(major_opcode), ctypes.byref(first_event), ctypes.byref(first_error))
    return bool(major_opcode.value)