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
class FFmpegDecoder(MediaDecoder):

    def get_file_extensions(self):
        return ('.mp3', '.ogg')

    def decode(self, filename, file, streaming=True):
        if streaming:
            return FFmpegSource(filename, file)
        else:
            return StaticSource(FFmpegSource(filename, file))