import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class PyOggSource(StreamingSource):

    def __init__(self, filename, file):
        self.filename = filename
        self.file = file
        self._stream = None
        self.sample_size = 16
        self._load_source()
        self.audio_format = AudioFormat(channels=self._stream.channels, sample_size=self.sample_size, sample_rate=self._stream.frequency)

    @abstractmethod
    def _load_source(self):
        pass

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        """Data returns as c_short_array instead of LP_c_char or c_ubyte, cast each buffer."""
        data = self._stream.get_buffer()
        if data is not None:
            buff, length = data
            buff_char_p = cast(buff, POINTER(c_char))
            return AudioData(buff_char_p[:length], length, 1000, 1000, [])
        return None

    def __del__(self):
        if self._stream:
            del self._stream