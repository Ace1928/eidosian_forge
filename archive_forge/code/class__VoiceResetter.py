from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
class _VoiceResetter:
    """Manage a voice during its reset period."""

    def __init__(self, driver, voice, voice_key, remaining_data) -> None:
        self.driver = driver
        self.voice = voice
        self.voice_key = voice_key
        self.remaining_data = remaining_data

    def run(self):
        if self.voice.buffers_queued != 0:
            self.voice._callback.on_buffer_end = self.flush_on_buffer_end
            self.voice.flush()
        else:
            pyglet.clock.schedule_once(self._finish, 0)

    def flush_on_buffer_end(self, *_):
        if self.voice.buffers_queued == 0:
            self.remaining_data.clear()
            pyglet.clock.schedule_once(self._finish, 0)

    def _finish(self, *_):
        self.voice._callback.on_buffer_end = None
        self.voice.samples_played_at_last_recycle = self.voice.samples_played
        self.driver._return_reset_voice(self.voice, self.voice_key)

    def destroy(self):
        pyglet.clock.unschedule(self._finish)
        self.driver = None
        self.voice = None
        self.remaining_data.clear()