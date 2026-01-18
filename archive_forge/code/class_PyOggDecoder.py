import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class PyOggDecoder(MediaDecoder):
    vorbis_exts = ('.ogg',) if pyogg.PYOGG_OGG_AVAIL and pyogg.PYOGG_VORBIS_AVAIL and pyogg.PYOGG_VORBIS_FILE_AVAIL else ()
    flac_exts = ('.flac',) if pyogg.PYOGG_FLAC_AVAIL else ()
    opus_exts = ('.opus',) if pyogg.PYOGG_OPUS_AVAIL and pyogg.PYOGG_OPUS_FILE_AVAIL else ()
    exts = vorbis_exts + flac_exts + opus_exts

    def get_file_extensions(self):
        return PyOggDecoder.exts

    def decode(self, filename, file, streaming=True):
        name, ext = os.path.splitext(filename)
        if ext in PyOggDecoder.vorbis_exts:
            source = PyOggVorbisSource
        elif ext in PyOggDecoder.flac_exts:
            source = PyOggFLACSource
        elif ext in PyOggDecoder.opus_exts:
            source = PyOggOpusSource
        else:
            raise DecodeException('Decoder could not find a suitable source to use with this filetype.')
        if streaming:
            return source(filename, file)
        else:
            return StaticSource(source(filename, file))