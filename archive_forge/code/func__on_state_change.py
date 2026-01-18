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
@staticmethod
def _on_state_change(device):
    for button, name in controller_api_to_pyglet.items():
        device.controls[name].value = device.xinput_state.Gamepad.wButtons & button
    device.controls['lefttrigger'].value = device.xinput_state.Gamepad.bLeftTrigger
    device.controls['righttrigger'].value = device.xinput_state.Gamepad.bRightTrigger
    device.controls['leftx'].value = device.xinput_state.Gamepad.sThumbLX
    device.controls['lefty'].value = device.xinput_state.Gamepad.sThumbLY
    device.controls['rightx'].value = device.xinput_state.Gamepad.sThumbRX
    device.controls['righty'].value = device.xinput_state.Gamepad.sThumbRY
    device.packet_number = device.xinput_state.dwPacketNumber