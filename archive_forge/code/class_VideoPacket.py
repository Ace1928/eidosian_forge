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
class VideoPacket(_Packet):
    _next_id = 0

    def __init__(self, packet, timestamp):
        super(VideoPacket, self).__init__(packet, timestamp)
        self.image = 0
        self.id = self._next_id
        VideoPacket._next_id += 1