import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class MemoryVorbisFileStream(UnclosedVorbisFileStream):

    def __init__(self, path, file):
        buff = create_string_buffer(pyogg.PYOGG_STREAM_BUFFER_SIZE)
        self.vf = pyogg.vorbis.OggVorbis_File()
        self.memory_object = MemoryVorbisObject(file)
        error = pyogg.vorbis.libvorbisfile.ov_open_callbacks(buff, self.vf, None, 0, self.memory_object.callbacks)
        if error != 0:
            raise DecodeException("file couldn't be opened or doesn't exist. Error code : {}".format(error))
        info = pyogg.vorbis.ov_info(byref(self.vf), -1)
        self.channels = info.contents.channels
        self.frequency = info.contents.rate
        array = (c_char * (pyogg.PYOGG_STREAM_BUFFER_SIZE * self.channels))()
        self.buffer_ = cast(pointer(array), c_char_p)
        self.bitstream = c_int()
        self.bitstream_pointer = pointer(self.bitstream)
        self.exists = True