import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class MemoryOpusObject:

    def __init__(self, filename, file):
        self.file = file
        self.filename = filename

        def read_func_cb(stream, buffer, size):
            data = self.file.read(size)
            read_size = len(data)
            memmove(buffer, data, read_size)
            return read_size

        def seek_func_cb(stream, offset, whence):
            self.file.seek(offset, whence)
            return 0

        def tell_func_cb(stream):
            pos = self.file.tell()
            return pos

        def close_func_cb(stream):
            return 0
        self.read_func = pyogg.opus.op_read_func(read_func_cb)
        self.seek_func = pyogg.opus.op_seek_func(seek_func_cb)
        self.tell_func = pyogg.opus.op_tell_func(tell_func_cb)
        self.close_func = pyogg.opus.op_close_func(close_func_cb)
        self.callbacks = pyogg.opus.OpusFileCallbacks(self.read_func, self.seek_func, self.tell_func, self.close_func)