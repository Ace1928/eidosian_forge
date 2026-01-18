import sys
from collections import deque
from ctypes import (c_int, c_int32, c_uint8, c_char_p,
import pyglet
import pyglet.lib
from pyglet import image
from pyglet.util import asbytes, asstr
from . import MediaDecoder
from .base import AudioData, SourceInfo, StaticSource
from .base import StreamingSource, VideoFormat, AudioFormat
from .ffmpeg_lib import *
from ..exceptions import MediaFormatException
def _get_audio_packet(self):
    """Take an audio packet from the queue.

        This function will schedule its `_fillq` function to fill up
        the queues if space is available. Multiple calls to this method will
        only result in one scheduled call to `_fillq`.
        """
    audio_data = self.audioq.popleft()
    low_lvl = self._check_low_level()
    if not low_lvl and (not self._fillq_scheduled):
        pyglet.clock.schedule_once(lambda dt: self._fillq(), 0)
        self._fillq_scheduled = True
    return audio_data