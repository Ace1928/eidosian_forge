import sys
import warnings
from ctypes import CFUNCTYPE, byref, c_void_p, c_int, c_ubyte, c_bool, c_uint32, c_uint64
import pyglet
from pyglet.event import EventDispatcher
from pyglet.input.base import Device, AbsoluteAxis, RelativeAxis, Button
from pyglet.input.base import Joystick, Controller, AppleRemote, ControllerManager
from pyglet.input.controller import get_mapping, create_guid
from pyglet.libs.darwin.cocoapy import CFSTR, CFIndex, CFTypeID, known_cftypes
from pyglet.libs.darwin.cocoapy import kCFRunLoopDefaultMode, CFAllocatorRef, cf
from pyglet.libs.darwin.cocoapy import cfset_to_set, cftype_to_value, cfarray_to_list
def _set_initial_control_values(self):
    for element in self.device.elements:
        if element.cookie in self._controls:
            control = self._controls[element.cookie]
            hid_value = self.device.get_value(element)
            if hid_value:
                control.value = hid_value.intvalue