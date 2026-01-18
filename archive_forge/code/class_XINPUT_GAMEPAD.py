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
class XINPUT_GAMEPAD(Structure):
    _fields_ = [('wButtons', WORD), ('bLeftTrigger', BYTE), ('bRightTrigger', BYTE), ('sThumbLX', SHORT), ('sThumbLY', SHORT), ('sThumbRX', SHORT), ('sThumbRY', SHORT)]