import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class PyOggOpusSource(PyOggSource):

    def _load_source(self):
        if self.file:
            self._stream = MemoryOpusFileStream(self.filename, self.file)
        else:
            self._stream = UnclosedOpusFileStream(self.filename)
        self._duration = self._stream.pcm_size / self._stream.frequency
        self._duration_per_frame = self._duration / self._stream.pcm_size

    def seek(self, timestamp):
        timestamp = max(0.0, min(timestamp, self._duration))
        position = int(timestamp / self._duration_per_frame)
        error = pyogg.opus.op_pcm_seek(self._stream.of, position)
        if error:
            warnings.warn(f'Opus stream could not seek properly {error}.')