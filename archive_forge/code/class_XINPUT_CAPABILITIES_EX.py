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
class XINPUT_CAPABILITIES_EX(Structure):
    _fields_ = [('Capabilities', XINPUT_CAPABILITIES), ('vendorId', WORD), ('productId', WORD), ('revisionId', WORD), ('a4', DWORD)]