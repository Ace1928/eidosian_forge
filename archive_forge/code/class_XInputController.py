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
class XInputController(Controller):

    def _initialize_controls(self):
        for button_name in controller_api_to_pyglet.values():
            control = self.device.controls[button_name]
            self._button_controls.append(control)
            self._add_button(control, button_name)
        for axis_name in ('leftx', 'lefty', 'rightx', 'righty', 'lefttrigger', 'righttrigger'):
            control = self.device.controls[axis_name]
            self._axis_controls.append(control)
            self._add_axis(control, axis_name)

    def _add_axis(self, control, name):
        tscale = 1.0 / (control.max - control.min)
        scale = 2.0 / (control.max - control.min)
        bias = -1.0 - control.min * scale
        if name in ('lefttrigger', 'righttrigger'):

            @control.event
            def on_change(value):
                normalized_value = value * tscale
                setattr(self, name, normalized_value)
                self.dispatch_event('on_trigger_motion', self, name, normalized_value)
        elif name in ('leftx', 'lefty'):

            @control.event
            def on_change(value):
                normalized_value = value * scale + bias
                setattr(self, name, normalized_value)
                self.dispatch_event('on_stick_motion', self, 'leftstick', self.leftx, self.lefty)
        elif name in ('rightx', 'righty'):

            @control.event
            def on_change(value):
                normalized_value = value * scale + bias
                setattr(self, name, normalized_value)
                self.dispatch_event('on_stick_motion', self, 'rightstick', self.rightx, self.righty)

    def _add_button(self, control, name):
        if name in ('dpleft', 'dpright', 'dpup', 'dpdown'):

            @control.event
            def on_change(value):
                setattr(self, name, value)
                self.dispatch_event('on_dpad_motion', self, self.dpleft, self.dpright, self.dpup, self.dpdown)
        else:

            @control.event
            def on_change(value):
                setattr(self, name, value)

            @control.event
            def on_press():
                self.dispatch_event('on_button_press', self, name)

            @control.event
            def on_release():
                self.dispatch_event('on_button_release', self, name)

    def rumble_play_weak(self, strength=1.0, duration=0.5):
        self.device.vibration.wRightMotorSpeed = int(max(min(1.0, strength), 0) * 65535)
        self.device.weak_duration = duration
        self.device.set_rumble_state()

    def rumble_play_strong(self, strength=1.0, duration=0.5):
        self.device.vibration.wLeftMotorSpeed = int(max(min(1.0, strength), 0) * 65535)
        self.device.strong_duration = duration
        self.device.set_rumble_state()

    def rumble_stop_weak(self):
        self.device.vibration.wRightMotorSpeed = 0
        self.device.set_rumble_state()

    def rumble_stop_strong(self):
        self.device.vibration.wLeftMotorSpeed = 0
        self.device.set_rumble_state()