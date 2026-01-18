import os
import time
import fcntl
import ctypes
import warnings
from ctypes import c_uint16 as _u16
from ctypes import c_int16 as _s16
from ctypes import c_uint32 as _u32
from ctypes import c_int32 as _s32
from ctypes import c_int64 as _s64
from concurrent.futures import ThreadPoolExecutor
from typing import List
import pyglet
from .evdev_constants import *
from pyglet.app.xlib import XlibSelectDevice
from pyglet.input.base import Device, RelativeAxis, AbsoluteAxis, Button, Joystick, Controller
from pyglet.input.base import DeviceOpenException, ControllerManager
from pyglet.input.controller import get_mapping, Relation, create_guid
def _make_device(self, name, count=1):
    path = os.path.join('/dev/input', name)
    while count > 0:
        try:
            return (name, EvdevDevice(self._display, path))
        except OSError:
            if count > 0:
                time.sleep(0.1)
            count -= 1
    return (None, None)