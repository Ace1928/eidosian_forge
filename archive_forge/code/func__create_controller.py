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
def _create_controller(device):
    mapping = get_mapping(device.get_guid())
    if device._type in (dinput.DI8DEVTYPE_JOYSTICK, dinput.DI8DEVTYPE_1STPERSON, dinput.DI8DEVTYPE_GAMEPAD):
        if mapping is not None:
            return base.Controller(device, mapping)
        else:
            warnings.warn(f'Warning: {device} (GUID: {device.get_guid()}) has no controller mappings. Update the mappings in the Controller DB.')