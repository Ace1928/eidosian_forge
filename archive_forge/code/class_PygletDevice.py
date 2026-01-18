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
class PygletDevice(Device):

    def __init__(self, display, device):
        super().__init__(display=display, name=device.product)
        self.device = device
        self.device.add_value_observer(self)
        self._create_controls()

    def open(self, window=None, exclusive=False):
        super().open(window, exclusive)
        self.device.open(exclusive)
        self._set_initial_control_values()

    def close(self):
        super().close()
        self.device.close()

    def get_controls(self):
        return list(self._controls.values())

    def get_guid(self):
        return self.device.get_guid()

    def device_value_changed(self, hid_device, hid_value):
        control = self._controls[hid_value.element.cookie]
        control.value = hid_value.intvalue

    def _create_controls(self):
        controls = []
        for element in self.device.elements:
            raw_name = element.name or '0x%x:%x' % (element.usagePage, element.usage)
            if element.type in (kIOHIDElementTypeInput_Misc, kIOHIDElementTypeInput_Axis):
                name = _axis_names.get((element.usagePage, element.usage))
                if element.isRelative:
                    control = RelativeAxis(name, raw_name)
                else:
                    control = AbsoluteAxis(name, element.logicalMin, element.logicalMax, raw_name)
            elif element.type == kIOHIDElementTypeInput_Button:
                name = _button_names.get((element.usagePage, element.usage))
                control = Button(name, raw_name)
            else:
                continue
            control._cookie = element.cookie
            control._usage = element.usage
            controls.append(control)
        controls.sort(key=lambda c: c._usage)
        self._controls = {control._cookie: control for control in controls}

    def _set_initial_control_values(self):
        for element in self.device.elements:
            if element.cookie in self._controls:
                control = self._controls[element.cookie]
                hid_value = self.device.get_value(element)
                if hid_value:
                    control.value = hid_value.intvalue

    def __repr__(self):
        return f'{self.__class__.__name__}({self.device})'