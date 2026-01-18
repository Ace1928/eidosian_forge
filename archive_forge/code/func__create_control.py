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
def _create_control(object_instance):
    raw_name = object_instance.tszName
    ctrl_type = object_instance.dwType
    instance = dinput.DIDFT_GETINSTANCE(ctrl_type)
    if ctrl_type & dinput.DIDFT_ABSAXIS:
        name = _abs_instance_names.get(instance)
        control = base.AbsoluteAxis(name, 0, 65535, raw_name)
    elif ctrl_type & dinput.DIDFT_RELAXIS:
        name = _rel_instance_names.get(instance)
        control = base.RelativeAxis(name, raw_name)
    elif ctrl_type & dinput.DIDFT_BUTTON:
        name = _btn_instance_names.get(instance)
        control = base.Button(name, raw_name)
    elif ctrl_type & dinput.DIDFT_POV:
        control = base.AbsoluteAxis(base.AbsoluteAxis.HAT, 0, 4294967295, raw_name)
    else:
        return
    control._type = object_instance.dwType
    return control