from collections import namedtuple, defaultdict
import threading
import weakref
from pyglet.media.devices.base import DeviceFlow
import pyglet
from pyglet.libs.win32.types import *
from pyglet.util import debug_print
from pyglet.media.devices import get_audio_device_manager
from . import lib_xaudio2 as lib
def _shutdown_xaudio2(self):
    """Stops and destroys all active voices, then destroys XA2 instance."""
    for player in self._in_use.values():
        player.on_driver_destroy()
        self._players.append(player.player)
    self._delete_driver()