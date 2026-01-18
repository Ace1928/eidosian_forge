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
def _device_enum(device_instance, arg):
    guid_id = format(device_instance.contents.guidProduct.Data1, '08x')
    if guid_id in _xinput_devices:
        return dinput.DIENUM_CONTINUE
    for dev in list(_missing_devices):
        if dev.matches(guid_id, device_instance):
            _missing_devices.remove(dev)
            return dinput.DIENUM_CONTINUE
    device = dinput.IDirectInputDevice8()
    _i_dinput.CreateDevice(device_instance.contents.guidInstance, ctypes.byref(device), None)
    di_dev = DirectInputDevice(display, device, device_instance.contents)
    _new_devices.append(di_dev)
    return dinput.DIENUM_CONTINUE