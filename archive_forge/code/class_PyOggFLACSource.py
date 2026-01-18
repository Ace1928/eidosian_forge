import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class PyOggFLACSource(PyOggSource):

    def _load_source(self):
        if self.file:
            self._stream = MemoryFLACFileStream(self.filename, self.file)
        else:
            self._stream = UnclosedFLACFileStream(self.filename)
        self.sample_size = self._stream.bits_per_sample
        self._duration = self._stream.total_samples / self._stream.frequency
        if self._stream.total_samples == 0:
            if _debug:
                warnings.warn(f'Unknown amount of samples found in {self.filename}. Seeking may be limited.')
            self._duration_per_frame = 0
        else:
            self._duration_per_frame = self._duration / self._stream.total_samples

    def seek(self, timestamp):
        if self._stream.seekable:
            if self._duration_per_frame:
                timestamp = max(0.0, min(timestamp, self._duration))
                position = int(timestamp / self._duration_per_frame)
            else:
                position = 0
            seek_succeeded = pyogg.flac.FLAC__stream_decoder_seek_absolute(self._stream.decoder, position)
            if seek_succeeded is False:
                warnings.warn(f'Failed to seek FLAC file: {self.filename}')
        else:
            warnings.warn(f'Stream is not seekable for FLAC file: {self.filename}.')