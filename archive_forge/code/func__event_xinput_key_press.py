import ctypes
import pyglet
from pyglet.input.base import Device, DeviceOpenException
from pyglet.input.base import Button, RelativeAxis, AbsoluteAxis
from pyglet.libs.x11 import xlib
from pyglet.util import asstr
@pyglet.window.xlib.XlibEventHandler(0)
def _event_xinput_key_press(self, ev):
    e = ctypes.cast(ctypes.byref(ev), ctypes.POINTER(xi.XDeviceKeyEvent)).contents
    device = self._responders.get(e.deviceid)
    if device is not None:
        device._key_press(e)