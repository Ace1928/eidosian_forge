from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
class XA2EngineCallback(com.COMObject):
    _interfaces_ = [lib.IXAudio2EngineCallback]

    def __init__(self, lock):
        super().__init__()
        self._lock = lock

    def OnProcessingPassStart(self):
        self._lock.acquire()

    def OnProcessingPassEnd(self):
        self._lock.release()

    def OnCriticalError(self, hresult):
        if self._lock.locked():
            self._lock.release()
        raise Exception('Critical Error:', hresult)