from collections import deque
import ctypes
import weakref
from abc import ABCMeta, abstractmethod
import pyglet
from pyglet.media.codecs import AudioData
from pyglet.util import debug_print
def _raw_play_cursor_to_time(self, cursor):
    if cursor is None:
        return None
    return self._to_perceived_play_cursor(cursor) / self.source.audio_format.bytes_per_second