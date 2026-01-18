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
def _maybe_fill_audio_data_buffer(self) -> None:
    if self._pyglet_source_exhausted:
        return
    refill_size = self._audio_data_buffer.get_ideal_refill_size(self._pending_bytes)
    if refill_size == 0:
        return
    self._audio_data_lock.release()
    refill_size = self.source.audio_format.align(refill_size)
    assert _debug(f'PulseAudioPlayer: Getting {refill_size}B of audio data')
    new_data = self._get_and_compensate_audio_data(refill_size, self._get_read_index())
    self._audio_data_lock.acquire()
    if new_data is None:
        self._pyglet_source_exhausted = True
        if self._has_underrun:
            MediaEvent('on_eos').sync_dispatch_to_player(self.player)
    else:
        self._audio_data_buffer.add_data(new_data)
        self.append_events(self._audio_data_buffer.virtual_write_index, new_data.events)