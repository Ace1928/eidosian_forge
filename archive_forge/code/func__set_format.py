import ctypes
import warnings
from typing import List, Dict, Optional
from pyglet.libs.win32.constants import WM_DEVICECHANGE, DBT_DEVICEARRIVAL, DBT_DEVICEREMOVECOMPLETE, \
from pyglet.event import EventDispatcher
import pyglet
from pyglet.input import base
from pyglet.libs import win32
from pyglet.libs.win32 import dinput, _user32, DEV_BROADCAST_DEVICEINTERFACE, com, DEV_BROADCAST_HDR
from pyglet.libs.win32 import _kernel32
from pyglet.input.controller import get_mapping
from pyglet.input.base import ControllerManager
def _set_format(self):
    if not self.controls:
        return
    object_formats = (dinput.DIOBJECTDATAFORMAT * len(self.controls))()
    offset = 0
    for object_format, control in zip(object_formats, self.controls):
        object_format.dwOfs = offset
        object_format.dwType = control._type
        offset += 4
    fmt = dinput.DIDATAFORMAT()
    fmt.dwSize = ctypes.sizeof(fmt)
    fmt.dwObjSize = ctypes.sizeof(dinput.DIOBJECTDATAFORMAT)
    fmt.dwFlags = 0
    fmt.dwDataSize = offset
    fmt.dwNumObjs = len(object_formats)
    fmt.rgodf = ctypes.cast(ctypes.pointer(object_formats), dinput.LPDIOBJECTDATAFORMAT)
    self._device.SetDataFormat(fmt)
    prop = dinput.DIPROPDWORD()
    prop.diph.dwSize = ctypes.sizeof(prop)
    prop.diph.dwHeaderSize = ctypes.sizeof(prop.diph)
    prop.diph.dwObj = 0
    prop.diph.dwHow = dinput.DIPH_DEVICE
    prop.dwData = 64 * ctypes.sizeof(dinput.DIDATAFORMAT)
    self._device.SetProperty(dinput.DIPROP_BUFFERSIZE, ctypes.byref(prop.diph))