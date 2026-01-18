from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def _create_new_voice(self, audio_format):
    """Has the driver create a new source voice for the given audio format."""
    voice = lib.IXAudio2SourceVoice()
    wfx_format = create_xa2_waveformat(audio_format)
    callback = XAudio2VoiceCallback()
    self._xaudio2.CreateSourceVoice(ctypes.byref(voice), ctypes.byref(wfx_format), 0, self.max_frequency_ratio, callback, None, None)
    return XA2SourceVoice(voice, callback, audio_format.channels, audio_format.sample_size)