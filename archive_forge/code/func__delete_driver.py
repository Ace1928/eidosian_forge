from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def _delete_driver(self):
    if self._xaudio2:
        assert _debug('XAudio2Driver: Deleting')
        if self.allow_3d:
            pyglet.clock.unschedule(self._calculate_3d_sources)
        self._destroy_voices()
        self._xaudio2.UnregisterForCallbacks(self._engine_callback)
        self._xaudio2.StopEngine()
        self._xaudio2.Release()
        self._xaudio2 = None