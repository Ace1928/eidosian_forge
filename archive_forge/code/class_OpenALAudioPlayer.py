from collections import deque
from typing import TYPE_CHECKING, List, Optional, Tuple
import weakref
from pyglet.media.drivers.base import AbstractAudioDriver, AbstractAudioPlayer, MediaEvent
from pyglet.media.drivers.listener import AbstractListener
from pyglet.media.drivers.openal import interface
from pyglet.media.player_worker_thread import PlayerWorkerThread
from pyglet.util import debug_print
class OpenALAudioPlayer(AbstractAudioPlayer):

    def __init__(self, driver: 'OpenALDriver', source: 'Source', player: 'Player') -> None:
        super().__init__(source, player)
        self.driver = driver
        self.alsource = driver.context.create_source()
        self._buffer_cursor = 0
        self._play_cursor = 0
        self._write_cursor = 0
        self._pyglet_source_exhausted = False
        self._has_underrun = False
        self._queued_buffer_sizes = deque()

    def delete(self) -> None:
        if self.alsource is not None:
            self.driver.worker.remove(self)
            self.alsource.delete()
            self.alsource = None

    def play(self) -> None:
        assert _debug('OpenALAudioPlayer.play()')
        assert self.driver is not None
        assert self.alsource is not None
        if not self.alsource.is_playing:
            self.alsource.play()
        self.driver.worker.add(self)

    def stop(self) -> None:
        assert _debug('OpenALAudioPlayer.stop()')
        assert self.driver is not None
        assert self.alsource is not None
        self.driver.worker.remove(self)
        self.alsource.pause()

    def clear(self) -> None:
        assert _debug('OpenALAudioPlayer.clear()')
        assert self.driver is not None
        assert self.alsource is not None
        super().clear()
        self.alsource.stop()
        self.alsource.clear()
        self._buffer_cursor = 0
        self._play_cursor = 0
        self._write_cursor = 0
        self._pyglet_source_exhausted = False
        self._has_underrun = False
        self._queued_buffer_sizes.clear()

    def _check_processed_buffers(self) -> None:
        buffers_processed = self.alsource.unqueue_buffers()
        for _ in range(buffers_processed):
            self._buffer_cursor += self._queued_buffer_sizes.popleft()

    def _update_play_cursor(self) -> None:
        self._play_cursor = self._buffer_cursor + self.alsource.byte_offset

    def work(self) -> None:
        self._check_processed_buffers()
        self._update_play_cursor()
        self.dispatch_media_events(self._play_cursor)
        if self._pyglet_source_exhausted:
            if not self._has_underrun and (not self.alsource.is_playing):
                self._has_underrun = True
                assert _debug('OpenALAudioPlayer: Dispatching eos')
                MediaEvent('on_eos').sync_dispatch_to_player(self.player)
            return
        refilled = self._maybe_refill()
        if refilled and (not self.alsource.is_playing):
            self.alsource.play()

    def _maybe_refill(self) -> bool:
        if self._pyglet_source_exhausted:
            return False
        remaining_bytes = self._write_cursor - self._play_cursor
        if remaining_bytes >= self._buffered_data_comfortable_limit:
            return False
        missing_bytes = self._buffered_data_ideal_size - remaining_bytes
        self._refill(self.source.audio_format.align_ceil(missing_bytes))
        return True

    def get_play_cursor(self) -> int:
        return self._play_cursor

    def _refill(self, refill_size) -> None:
        audio_data = self._get_and_compensate_audio_data(refill_size, self._play_cursor)
        if audio_data is None:
            self._pyglet_source_exhausted = True
            return
        self.append_events(self._write_cursor, audio_data.events)
        buf = self.alsource.get_buffer()
        buf.data(audio_data, self.source.audio_format)
        self.alsource.queue_buffer(buf)
        self._write_cursor += audio_data.length
        self._queued_buffer_sizes.append(audio_data.length)

    def prefill_audio(self) -> None:
        self._maybe_refill()

    def set_volume(self, volume: float) -> None:
        self.alsource.gain = volume

    def set_position(self, position: Tuple[float, float, float]) -> None:
        self.alsource.position = position

    def set_min_distance(self, min_distance: float) -> None:
        self.alsource.reference_distance = min_distance

    def set_max_distance(self, max_distance: float) -> None:
        self.alsource.max_distance = max_distance

    def set_pitch(self, pitch: float) -> None:
        self.alsource.pitch = pitch

    def set_cone_orientation(self, cone_orientation: Tuple[float, float, float]) -> None:
        self.alsource.direction = cone_orientation

    def set_cone_inner_angle(self, cone_inner_angle: float) -> None:
        self.alsource.cone_inner_angle = cone_inner_angle

    def set_cone_outer_angle(self, cone_outer_angle: float) -> None:
        self.alsource.cone_outer_angle = cone_outer_angle

    def set_cone_outer_gain(self, cone_outer_gain: float) -> None:
        self.alsource.cone_outer_gain = cone_outer_gain