from typing import Dict, Optional
import pyglet
from pyglet.input import base
from pyglet.input.win32.directinput import _di_manager as _di_device_manager
from pyglet.input.win32.directinput import DirectInputDevice, _create_controller
from pyglet.input.win32.directinput import get_devices as dinput_get_devices
from pyglet.input.win32.directinput import get_controllers as dinput_get_controllers
from pyglet.input.win32.directinput import get_joysticks
def _set_initial_didevices(self):
    if not _di_device_manager.registered:
        _di_device_manager.register_device_events()
        _di_device_manager.set_current_devices()
    for device in _di_device_manager.devices:
        self._add_di_controller(device)