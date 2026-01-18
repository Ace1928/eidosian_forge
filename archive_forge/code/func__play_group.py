from collections import deque
import ctypes
import weakref
from abc import ABCMeta, abstractmethod
import pyglet
from pyglet.media.codecs import AudioData
from pyglet.util import debug_print
def _play_group(self, audio_players):
    """Begin simultaneous playback on a list of audio players."""
    for player in audio_players:
        player.play()