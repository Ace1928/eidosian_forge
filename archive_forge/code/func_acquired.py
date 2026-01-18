from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def acquired(self, on_buffer_end_cb, sample_rate):
    """A voice has been acquired. Set the callback as well as its new sample
        rate.
        """
    self._callback.on_buffer_end = on_buffer_end_cb
    self._voice.SetSourceSampleRate(sample_rate)