import os
import fcntl
import ctypes
import warnings
from ctypes import c_int as _int
from ctypes import c_uint8 as _u8
from ctypes import c_uint16 as _u16
from ctypes import c_int16 as _s16
from ctypes import c_uint32 as _u32
from ctypes import c_int32 as _s32
from ctypes import c_int64 as _s64
from ctypes import create_string_buffer
from concurrent.futures import ThreadPoolExecutor
import pyglet
from pyglet.app.xlib import XlibSelectDevice
from pyglet.input.base import Device, RelativeAxis, AbsoluteAxis, Button, Joystick, Controller
from pyglet.input.base import DeviceOpenException, ControllerManager
from pyglet.input.linux.evdev_constants import *
from pyglet.input.controller import get_mapping, Relation, create_guid
class HIDRawReportDescriptor(ctypes.Structure):
    _fields_ = (('size', _u32), ('values', _u8 * 4096))