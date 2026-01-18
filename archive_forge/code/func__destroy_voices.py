from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def _destroy_voices(self):
    """Destroy and clear all voice pools."""
    for list_ in self._voice_pool.values():
        for voice in list_:
            voice.destroy()
        list_.clear()
    for voice, resetter in self._resetting_voices.items():
        voice.destroy()
        resetter.destroy()
    self._resetting_voices.clear()
    for voice in self.active_voices:
        voice.destroy()
    self._in_use.clear()