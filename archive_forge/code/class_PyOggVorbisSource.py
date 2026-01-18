import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class PyOggVorbisSource(PyOggSource):

    def _load_source(self):
        if self.file:
            self._stream = MemoryVorbisFileStream(self.filename, self.file)
        else:
            self._stream = UnclosedVorbisFileStream(self.filename)
        self._duration = pyogg.vorbis.libvorbisfile.ov_time_total(byref(self._stream.vf), -1)

    def get_audio_data(self, num_bytes, compensation_time=0.0):
        data = self._stream.get_buffer()
        if data is not None:
            return AudioData(*data, 1000, 1000, [])
        return None

    def seek(self, timestamp):
        seek_succeeded = pyogg.vorbis.ov_time_seek(self._stream.vf, timestamp)
        if seek_succeeded != 0:
            if _debug:
                warnings.warn(f'Failed to seek file {self.filename} - {seek_succeeded}')