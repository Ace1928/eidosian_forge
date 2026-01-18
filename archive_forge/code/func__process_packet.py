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
def _process_packet(self):
    """Process the packet that has been just read.

        Determines whether it's a video or audio packet and queue it in the
        appropriate queue.
        """
    timestamp = ffmpeg_get_packet_pts(self._file, self._packet)
    timestamp = timestamp_from_ffmpeg(timestamp)
    timestamp -= self.start_time
    if self._packet.contents.stream_index == self._video_stream_index:
        video_packet = VideoPacket(self._packet, timestamp)
        if _debug:
            print('Created and queued packet %d (%f)' % (video_packet.id, video_packet.timestamp))
        self.videoq.append(video_packet)
        return video_packet
    elif self.audio_format and self._packet.contents.stream_index == self._audio_stream_index:
        audio_packet = AudioPacket(self._packet, timestamp)
        self.audioq.append(audio_packet)
        return audio_packet