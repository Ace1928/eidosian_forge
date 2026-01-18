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
class XInputDevice(Device):

    def __init__(self, index, manager):
        super().__init__(None, f'XInput{index}')
        self.index = index
        self._manager = weakref.proxy(manager)
        self.connected = False
        self.xinput_state = XINPUT_STATE()
        self.packet_number = 0
        self.vibration = XINPUT_VIBRATION()
        self.weak_duration = None
        self.strong_duration = None
        self.controls = {'a': Button('a'), 'b': Button('b'), 'x': Button('x'), 'y': Button('y'), 'back': Button('back'), 'start': Button('start'), 'guide': Button('guide'), 'leftshoulder': Button('leftshoulder'), 'rightshoulder': Button('rightshoulder'), 'leftstick': Button('leftstick'), 'rightstick': Button('rightstick'), 'dpup': Button('dpup'), 'dpdown': Button('dpdown'), 'dpleft': Button('dpleft'), 'dpright': Button('dpright'), 'leftx': AbsoluteAxis('leftx', -32768, 32768), 'lefty': AbsoluteAxis('lefty', -32768, 32768), 'rightx': AbsoluteAxis('rightx', -32768, 32768), 'righty': AbsoluteAxis('righty', -32768, 32768), 'lefttrigger': AbsoluteAxis('lefttrigger', 0, 255), 'righttrigger': AbsoluteAxis('righttrigger', 0, 255)}

    def set_rumble_state(self):
        XInputSetState(self.index, byref(self.vibration))

    def get_controls(self):
        return list(self.controls.values())

    def get_guid(self):
        return 'XINPUTCONTROLLER'