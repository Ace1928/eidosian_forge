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
def _check_low_level(self):
    """Check if both audio and video queues are getting very low.

        If one of them has less than 2 elements, we fill the queue immediately
        with new packets. We don't wait for a scheduled call because we need
        them immediately.

        This would normally happens only during seek operations where we
        consume many packets to find the correct timestamp.
        """
    if len(self.audioq) < 2 or len(self.videoq) < 2:
        if len(self.audioq) < self._max_len_audioq and len(self.videoq) < self._max_len_videoq:
            self._fillq()
        return True
    return False