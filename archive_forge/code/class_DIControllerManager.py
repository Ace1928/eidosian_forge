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
class DIControllerManager(ControllerManager):

    def __init__(self, display=None):
        self._display = display
        self._controllers: Dict[DirectInputDevice, base.Controller] = {}
        for device in _di_manager.devices:
            self._add_controller(device)

        @_di_manager.event
        def on_connect(di_device):
            if di_device not in self._controllers:
                if self._add_controller(di_device):
                    pyglet.app.platform_event_loop.post_event(self, 'on_connect', self._controllers[di_device])

        @_di_manager.event
        def on_disconnect(di_device):
            if di_device in self._controllers:
                _controller = self._controllers[di_device]
                del self._controllers[di_device]
                pyglet.app.platform_event_loop.post_event(self, 'on_disconnect', _controller)

    def _add_controller(self, device: DirectInputDevice) -> Optional[base.Controller]:
        controller = _create_controller(device)
        if controller:
            self._controllers[device] = controller
            return controller
        return None

    def get_controllers(self):
        if not _di_manager.registered:
            _di_manager.register_device_events()
            _di_manager.set_current_devices()
        return list(self._controllers.values())