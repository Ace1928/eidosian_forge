from collections import deque
import ctypes
import threading
from typing import Deque, Optional, TYPE_CHECKING
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
from . import lib_pulseaudio as pa
from .interface import PulseAudioMainloop
class PulseAudioDriver(AbstractAudioDriver):

    def __init__(self) -> None:
        self.mainloop = PulseAudioMainloop()
        self.mainloop.start()
        self.context = None
        self.worker = PlayerWorkerThread()
        self.worker.start()
        self._players = weakref.WeakSet()
        self._listener = PulseAudioListener(self)

    def create_audio_player(self, source: 'Source', player: 'Player') -> 'PulseAudioPlayer':
        assert self.context is not None
        player = PulseAudioPlayer(source, player, self)
        self._players.add(player)
        return player

    def connect(self, server: Optional[bytes]=None) -> None:
        """Connect to pulseaudio server.

        :Parameters:
            `server` : bytes
                Server to connect to, or ``None`` for the default local
                server (which may be spawned as a daemon if no server is
                found).
        """
        assert not self.context, 'Already connected'
        self.context = self.mainloop.create_context()
        self.context.connect(server)

    def dump_debug_info(self):
        print('Client version: ', pa.pa_get_library_version())
        print('Server:         ', self.context.server)
        print('Protocol:       ', self.context.protocol_version)
        print('Server protocol:', self.context.server_protocol_version)
        print('Local context:  ', self.context.is_local and 'Yes' or 'No')

    def delete(self) -> None:
        """Completely shut down pulseaudio client."""
        if self.mainloop is None:
            return
        self.worker.stop()
        with self.mainloop.lock:
            if self.context is not None:
                self.context.delete()
                self.context = None
        self.mainloop.delete()
        self.mainloop = None

    def get_listener(self) -> 'PulseAudioListener':
        return self._listener