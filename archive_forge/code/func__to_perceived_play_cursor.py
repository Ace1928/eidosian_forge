from collections import deque
import ctypes
import weakref
from abc import ABCMeta, abstractmethod
import pyglet
from pyglet.media.codecs import AudioData
from pyglet.util import debug_print
def _to_perceived_play_cursor(self, play_cursor):
    return play_cursor - self._compensated_bytes