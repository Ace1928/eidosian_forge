import math
import ctypes
from . import interface
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
def _db2gain(db):
    """Convert 100ths of dB to linear gain."""
    return math.pow(10.0, float(db) / 1000.0)