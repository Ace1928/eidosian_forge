import ctypes
import sys
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, TypeVar, Union
import weakref
from pyglet.media.drivers.pulse import lib_pulseaudio as pa
from pyglet.media.exceptions import MediaException
from pyglet.util import debug_print
class PulseAudioContext(PulseAudioMainloopChild):
    """Basic object for a connection to a PulseAudio server."""
    _state_name = {pa.PA_CONTEXT_UNCONNECTED: 'Unconnected', pa.PA_CONTEXT_CONNECTING: 'Connecting', pa.PA_CONTEXT_AUTHORIZING: 'Authorizing', pa.PA_CONTEXT_SETTING_NAME: 'Setting Name', pa.PA_CONTEXT_READY: 'Ready', pa.PA_CONTEXT_FAILED: 'Failed', pa.PA_CONTEXT_TERMINATED: 'Terminated'}

    def __init__(self, mainloop: PulseAudioMainloop, name: bytes) -> None:
        super().__init__(mainloop)
        ctx = pa.pa_context_new_with_proplist(mainloop._pa_mainloop_vtab, name, None)
        self.check_ptr_not_null(ctx)
        self._pa_context = ctx
        self.state = None
        self._set_state_callback(self._state_callback)

    def delete(self) -> None:
        """Completely shut down pulseaudio client. Will lock."""
        if self._pa_context is not None:
            with self.mainloop.lock:
                assert _debug('PulseAudioContext.delete')
                if self.is_ready:
                    pa.pa_context_disconnect(self._pa_context)
                    while self.state is not None and (not self.is_terminated):
                        self.mainloop.wait()
                self._set_state_callback(0)
                pa.pa_context_unref(self._pa_context)
            self._pa_context = None

    @property
    def is_ready(self) -> bool:
        return self.state == pa.PA_CONTEXT_READY

    @property
    def is_failed(self) -> bool:
        return self.state == pa.PA_CONTEXT_FAILED

    @property
    def is_terminated(self) -> bool:
        return self.state == pa.PA_CONTEXT_TERMINATED

    @property
    def server(self) -> Optional[str]:
        if self.is_ready:
            return get_ascii_str_or_none(pa.pa_context_get_server(self._pa_context))
        return None

    @property
    def protocol_version(self) -> Optional[str]:
        if self._pa_context is not None:
            return get_uint32_or_none(pa.pa_context_get_protocol_version(self._pa_context))
        return None

    @property
    def server_protocol_version(self) -> Optional[str]:
        if self._pa_context is not None:
            return get_uint32_or_none(pa.pa_context_get_server_protocol_version(self._pa_context))
        return None

    @property
    def is_local(self) -> Optional[bool]:
        if self._pa_context is not None:
            return get_bool_or_none(pa.pa_context_is_local(self._pa_context))
        return None

    def connect(self, server: Optional[bytes]=None) -> None:
        """Connect the context to a PulseAudio server.

        Will grab the mainloop lock.

        :Parameters:
            `server` : bytes
                Server to connect to, or ``None`` for the default local
                server (which may be spawned as a daemon if no server is
                found).
        """
        assert self._pa_context is not None
        self.state = None
        with self.mainloop.lock:
            self.check(pa.pa_context_connect(self._pa_context, server, 0, None))
            while not self.is_failed and (not self.is_ready):
                self.mainloop.wait()
            if self.is_failed:
                self.raise_error()

    def create_stream(self, audio_format: 'AudioFormat') -> 'PulseAudioStream':
        """
        Create a new audio stream.
        """
        assert self.is_ready
        return PulseAudioStream(self, audio_format)

    def set_input_volume(self, stream: 'PulseAudioStream', volume: float) -> 'PulseAudioOperation':
        """
        Set the volume for a stream.
        """
        cvolume = self._get_cvolume_from_linear(stream, volume)
        clump = PulseAudioContextSuccessCallbackLump(self)
        return PulseAudioOperation(clump, pa.pa_context_set_sink_input_volume(self._pa_context, stream.index, cvolume, clump.pa_callback, None))

    def _get_cvolume_from_linear(self, stream: 'PulseAudioStream', volume: float) -> pa.pa_cvolume:
        cvolume = pa.pa_cvolume()
        volume = pa.pa_sw_volume_from_linear(volume)
        pa.pa_cvolume_set(cvolume, stream.get_sample_spec().channels, volume)
        return cvolume

    def _set_state_callback(self, py_callback: Optional[Callable[['PulseAudioContext', Any], Any]]) -> None:
        if py_callback is None:
            self._pa_state_change_callback = None
        else:
            self._pa_state_change_callback = pa.pa_context_notify_cb_t(py_callback)
        pa.pa_context_set_state_callback(self._pa_context, self._pa_state_change_callback, None)

    def _state_callback(self, context: 'PulseAudioContext', _userdata) -> None:
        self.state = pa.pa_context_get_state(self._pa_context)
        assert _debug(f'PulseAudioContext: state changed to {self._state_name[self.state]}')
        self.mainloop.signal()

    def check(self, result: T) -> T:
        if result < 0:
            self.raise_error()
        return result

    def check_not_null(self, value: T) -> T:
        if value is None:
            self.raise_error()
        return value

    def check_ptr_not_null(self, value: T) -> T:
        if not value:
            self.raise_error()
        return value

    def raise_error(self) -> None:
        error = pa.pa_context_errno(self._pa_context)
        raise PulseAudioException(error, get_ascii_str_or_none(pa.pa_strerror(error)))