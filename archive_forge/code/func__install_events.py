import ctypes
import pyglet
from pyglet.input.base import Device, DeviceOpenException
from pyglet.input.base import Button, RelativeAxis, AbsoluteAxis
from pyglet.libs.x11 import xlib
from pyglet.util import asstr
def _install_events(self, window):
    dispatcher = XInputWindowEventDispatcher.get_dispatcher(window)
    dispatcher.open_device(self._device_id, self._device, self)