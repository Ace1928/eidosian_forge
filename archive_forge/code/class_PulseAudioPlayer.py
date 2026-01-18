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
class PulseAudioPlayer(AbstractAudioPlayer):

    def __init__(self, source: 'Source', player: 'Player', driver: 'PulseAudioDriver') -> None:
        super().__init__(source, player)
        self.driver = driver
        self._volume = 1.0
        audio_format = source.audio_format
        assert audio_format
        self._latest_timing_info = None
        self._last_clear_read_index = 0
        self._pyglet_source_exhausted = False
        self._pending_bytes = 0
        ideal_size = audio_format.align_ceil(self._buffered_data_ideal_size // 2)
        comf_limit = audio_format.align_ceil(self._buffered_data_comfortable_limit // 2)
        self._audio_data_buffer = _AudioDataBuffer(ideal_size, comf_limit)
        self._audio_data_lock = threading.Lock()
        self._has_underrun = False
        with driver.mainloop.lock:
            self.stream = driver.context.create_stream(audio_format)
            self.stream.set_write_callback(self._write_callback)
            self.stream.set_underflow_callback(self._underflow_callback)
            self.stream.connect_playback(ideal_size, comf_limit)
            assert self.stream.is_ready
        assert _debug('PulseAudioPlayer: __init__ finished')

    def _write_callback(self, _stream, nbytes: int, _userdata) -> None:
        assert _debug(f'PulseAudioPlayer: Write requested, {nbytes}B')
        assert self.source.audio_format.align(nbytes) == nbytes
        with self._audio_data_lock:
            if self._audio_data_buffer.available > 0:
                written = self._write_to_stream(nbytes)
                if (unfulfilled := (nbytes - written)) > 0:
                    self._pending_bytes = unfulfilled
            else:
                self._pending_bytes = nbytes
        self.stream.mainloop.signal()

    def _underflow_callback(self, _stream, _userdata) -> None:
        assert _debug('PulseAudioPlayer: underflow')
        with self._audio_data_lock:
            if self._pyglet_source_exhausted and self._audio_data_buffer.available == 0:
                MediaEvent('on_eos').sync_dispatch_to_player(self.player)
            self._has_underrun = True
        self.stream.mainloop.signal()

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

    def _write_to_stream(self, nbytes: int) -> int:
        data_ptr, bytes_accepted = self.stream.begin_write(nbytes)
        bytes_written = self._audio_data_buffer.memmove(data_ptr.value, bytes_accepted)
        if bytes_written == 0:
            self.stream.cancel_write()
        else:
            self.stream.write(data_ptr, bytes_written, pa.PA_SEEK_RELATIVE)
        assert _debug(f'PulseAudioPlayer: Wrote {bytes_written}/{nbytes}')
        return bytes_written

    def _update_and_get_timing_info(self) -> Optional[pa.pa_timing_info]:
        self.stream.update_timing_info().wait().delete()
        return self.stream.get_timing_info()

    def _maybe_write_pending(self) -> None:
        with self._audio_data_lock:
            if self._pending_bytes == 0 or self._audio_data_buffer.available == 0:
                return
            written = self._write_to_stream(self._pending_bytes)
            self._pending_bytes -= written
            if not self._has_underrun:
                return
            self._has_underrun = False
        self.stream.trigger().wait().delete()

    def work(self) -> None:
        with self.driver.mainloop.lock:
            self._maybe_write_pending()
            self._latest_timing_info = self._update_and_get_timing_info()
        self.dispatch_media_events(self._get_read_index())
        with self._audio_data_lock:
            self._maybe_fill_audio_data_buffer()
        with self.driver.mainloop.lock:
            self._maybe_write_pending()

    def delete(self) -> None:
        assert _debug('PulseAudioPlayer.delete')
        self.driver.worker.remove(self)
        if self.driver.mainloop is None:
            assert _debug('PulseAudioPlayer.delete: PulseAudioDriver already deleted.')
        else:
            with self.driver.mainloop.lock:
                self.stream.delete()
                self.stream = None

    def clear(self) -> None:
        assert _debug('PulseAudioPlayer.clear')
        super().clear()
        self._pyglet_source_exhausted = False
        self._audio_data_buffer.clear()
        self._has_underrun = False
        with self.stream.mainloop.lock:
            ti = self._update_and_get_timing_info()
            assert not ti.read_index_corrupt
            self._last_clear_read_index = ti.read_index
            self.stream.flush().wait().delete()
            self.stream.prebuf().wait().delete()

    def play(self) -> None:
        assert _debug('PulseAudioPlayer.play')
        with self.stream.mainloop.lock:
            if self.stream.is_corked():
                self.stream.resume().wait().delete()
            assert not self.stream.is_corked()
        self.driver.worker.add(self)

    def stop(self) -> None:
        assert _debug('PulseAudioPlayer.stop')
        self.driver.worker.remove(self)
        with self.stream.mainloop.lock:
            self.stream.pause().wait().delete()

    def get_play_cursor(self) -> int:
        return self._get_read_index()

    def _get_read_index(self) -> int:
        if (t_info := self._latest_timing_info) is None:
            return 0
        read_idx = t_info.read_index - self._last_clear_read_index
        assert _debug(f'_get_read_index -> {read_idx}')
        return read_idx

    def set_volume(self, volume: float) -> None:
        self._volume = volume
        if self.stream:
            driver = self.driver
            volume *= driver._listener._volume
            with driver.context.mainloop.lock:
                driver.context.set_input_volume(self.stream, volume).wait().delete()

    def set_pitch(self, pitch):
        with self.stream.mainloop.lock:
            sample_rate = self.stream.get_sample_spec().rate
            self.stream.update_sample_rate(int(pitch * sample_rate)).wait().delete()

    def prefill_audio(self):
        self.work()