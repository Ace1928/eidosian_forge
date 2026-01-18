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
class HIDDevice:

    @classmethod
    def get_device(cls, device_ref):
        if device_ref.value in _device_lookup:
            return _device_lookup[device_ref.value]
        else:
            device = HIDDevice(device_ref)
            return device

    def __init__(self, device_ref):
        _device_lookup[device_ref.value] = self
        self.device_ref = device_ref
        self.transport = self.get_property('Transport')
        self.vendorID = self.get_property('VendorID')
        self.vendorIDSource = self.get_property('VendorIDSource')
        self.productID = self.get_property('ProductID')
        self.versionNumber = self.get_property('VersionNumber')
        self.manufacturer = self.get_property('Manufacturer')
        self.product = self.get_property('Product')
        self.serialNumber = self.get_property('SerialNumber')
        self.locationID = self.get_property('LocationID')
        self.primaryUsage = self.get_property('PrimaryUsage')
        self.primaryUsagePage = self.get_property('PrimaryUsagePage')
        self.elements = self._get_elements()
        self.value_observers = set()
        self.value_callback = self._register_input_value_callback()

    def get_guid(self):
        """Generate an SDL2 style GUID from the product guid."""
        bustype = 3
        vendor = self.vendorID or 0
        product = self.productID or 0
        version = self.versionNumber or 0
        name = self.product or ''
        return create_guid(bustype, vendor, product, version, name, 0, 0)

    def get_property(self, name):
        cfname = CFSTR(name)
        cfvalue = c_void_p(iokit.IOHIDDeviceGetProperty(self.device_ref, cfname))
        cf.CFRelease(cfname)
        return cftype_to_value(cfvalue)

    def open(self, exclusive_mode=False):
        if exclusive_mode:
            options = kIOHIDOptionsTypeSeizeDevice
        else:
            options = kIOHIDOptionsTypeNone
        return bool(iokit.IOHIDDeviceOpen(self.device_ref, options))

    def close(self):
        return bool(iokit.IOHIDDeviceClose(self.device_ref, kIOHIDOptionsTypeNone))

    def schedule_with_run_loop(self):
        iokit.IOHIDDeviceScheduleWithRunLoop(self.device_ref, c_void_p(cf.CFRunLoopGetCurrent()), kCFRunLoopDefaultMode)

    def unschedule_from_run_loop(self):
        iokit.IOHIDDeviceUnscheduleFromRunLoop(self.device_ref, c_void_p(cf.CFRunLoopGetCurrent()), kCFRunLoopDefaultMode)

    def _get_elements(self):
        cfarray = c_void_p(iokit.IOHIDDeviceCopyMatchingElements(self.device_ref, None, 0))
        if not cfarray:
            return []
        elements = cfarray_to_list(cfarray)
        cf.CFRelease(cfarray)
        return elements

    def conforms_to(self, page, usage):
        return bool(iokit.IOHIDDeviceConformsTo(self.device_ref, page, usage))

    def is_pointer(self):
        return self.conforms_to(1, 1)

    def is_mouse(self):
        return self.conforms_to(1, 2)

    def is_joystick(self):
        return self.conforms_to(1, 4)

    def is_gamepad(self):
        return self.conforms_to(1, 5)

    def is_keyboard(self):
        return self.conforms_to(1, 6)

    def is_keypad(self):
        return self.conforms_to(1, 7)

    def is_multi_axis(self):
        return self.conforms_to(1, 8)

    def py_value_callback(self, context, result, sender, value):
        v = HIDValue(c_void_p(value))
        for x in self.value_observers:
            if hasattr(x, 'device_value_changed'):
                x.device_value_changed(self, v)

    def _register_input_value_callback(self):
        value_callback = HIDDeviceValueCallback(self.py_value_callback)
        iokit.IOHIDDeviceRegisterInputValueCallback(self.device_ref, value_callback, None)
        return value_callback

    def add_value_observer(self, observer):
        self.value_observers.add(observer)

    def get_value(self, element):
        value_ref = c_void_p()
        iokit.IOHIDDeviceGetValue(self.device_ref, element.element_ref, byref(value_ref))
        if value_ref:
            return HIDValue(value_ref)
        else:
            return None

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.product})'