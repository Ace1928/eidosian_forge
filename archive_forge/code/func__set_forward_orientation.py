from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.drivers.openal import interface
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
def _set_forward_orientation(self, orientation: Tuple[float, float, float]) -> None:
    self._al_listener.orientation = orientation + self._up_orientation
    self._forward_orientation = orientation