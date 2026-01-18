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
class EvdevControllerManager(ControllerManager, XlibSelectDevice):

    def __init__(self, display=None):
        super().__init__()
        self._display = display
        self._devices_file = open('/proc/bus/input/devices')
        self._device_names = self._get_device_names()
        self._controllers = {}
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
        for name in self._device_names:
            path = os.path.join('/dev/input', name)
            try:
                device = EvdevDevice(self._display, path)
            except OSError:
                continue
            controller = _create_controller(device)
            if controller:
                self._controllers[name] = controller
        pyglet.app.platform_event_loop.select_devices.add(self)

    def __del__(self):
        self._devices_file.close()

    def fileno(self):
        """Allow this class to be Selectable"""
        return self._devices_file.fileno()

    @staticmethod
    def _get_device_names():
        return {name for name in os.listdir('/dev/input') if name.startswith('event')}

    def _make_device_callback(self, future):
        name, device = future.result()
        if not device:
            return
        if name in self._controllers:
            controller = self._controllers.get(name)
        else:
            controller = _create_controller(device)
            self._controllers[name] = controller
        if controller:
            pyglet.app.platform_event_loop.post_event(self, 'on_connect', controller)

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

    def select(self):
        """Triggered whenever the devices_file changes."""
        new_device_files = self._get_device_names()
        appeared = new_device_files - self._device_names
        disappeared = self._device_names - new_device_files
        self._device_names = new_device_files
        for name in appeared:
            future = self._thread_pool.submit(self._make_device, name, count=10)
            future.add_done_callback(self._make_device_callback)
        for name in disappeared:
            controller = self._controllers.get(name)
            if controller:
                self.dispatch_event('on_disconnect', controller)

    def get_controllers(self) -> List[Controller]:
        return list(self._controllers.values())