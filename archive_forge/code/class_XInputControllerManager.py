import time
import weakref
import threading
import pyglet
from pyglet.libs.win32 import com
from pyglet.event import EventDispatcher
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import _ole32 as ole32, _oleaut32 as oleaut32
from pyglet.libs.win32.constants import CLSCTX_INPROC_SERVER
from pyglet.input.base import Device, Controller, Button, AbsoluteAxis, ControllerManager
class XInputControllerManager(ControllerManager):

    def __init__(self):
        self._controllers = {}
        for device in _device_manager.all_devices:
            meta = {'name': device.name, 'guid': 'XINPUTCONTROLLER'}
            self._controllers[device] = XInputController(device, meta)

        @_device_manager.event
        def on_connect(xdevice):
            self.dispatch_event('on_connect', self._controllers[xdevice])

        @_device_manager.event
        def on_disconnect(xdevice):
            self.dispatch_event('on_disconnect', self._controllers[xdevice])

    def get_controllers(self):
        return [ctlr for ctlr in self._controllers.values() if ctlr.device.connected]