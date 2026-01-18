import pyogg
import os.path
import warnings
from abc import abstractmethod
from ctypes import c_void_p, POINTER, c_int, pointer, cast, c_char, c_char_p, CFUNCTYPE, c_ubyte
from ctypes import memmove, create_string_buffer, byref
from pyglet.media import StreamingSource
from pyglet.media.codecs import AudioFormat, AudioData, MediaDecoder, StaticSource
from pyglet.util import debug_print, DecodeException
class MemoryFLACFileStream(UnclosedFLACFileStream):

    def __init__(self, path, file):
        self.file = file
        self.file_size = 0
        if getattr(self.file, 'seek', None) and getattr(self.file, 'tell', None):
            self.seekable = True
            self.file.seek(0, 2)
            self.file_size = self.file.tell()
            self.file.seek(0)
        else:
            warnings.warn(f'Warning: {file} file object is not seekable.')
            self.seekable = False
        self.decoder = pyogg.flac.FLAC__stream_decoder_new()
        self.client_data = c_void_p()
        self.channels = None
        self.frequency = None
        self.total_samples = None
        self.buffer = None
        self.bytes_written = None
        self.write_callback_ = pyogg.flac.FLAC__StreamDecoderWriteCallback(self.write_callback)
        self.metadata_callback_ = pyogg.flac.FLAC__StreamDecoderMetadataCallback(self.metadata_callback)
        self.error_callback_ = pyogg.flac.FLAC__StreamDecoderErrorCallback(self.error_callback)
        self.read_callback_ = pyogg.flac.FLAC__StreamDecoderReadCallback(self.read_callback)
        if self.seekable:
            self.seek_callback_ = pyogg.flac.FLAC__StreamDecoderSeekCallback(self.seek_callback)
            self.tell_callback_ = pyogg.flac.FLAC__StreamDecoderTellCallback(self.tell_callback)
            self.length_callback_ = pyogg.flac.FLAC__StreamDecoderLengthCallback(self.length_callback)
            self.eof_callback_ = FLAC__StreamDecoderEofCallback(self.eof_callback)
        else:
            self.seek_callback_ = None
            self.tell_callback_ = None
            self.length_callback_ = None
            self.eof_callback_ = None
        init_status = pyogg.flac.libflac.FLAC__stream_decoder_init_stream(self.decoder, self.read_callback_, self.seek_callback_, self.tell_callback_, self.length_callback_, self.eof_callback_, self.write_callback_, self.metadata_callback_, self.error_callback_, self.client_data)
        if init_status:
            raise DecodeException("An error occurred when trying to open '{}': {}".format(path, pyogg.flac.FLAC__StreamDecoderInitStatusEnum[init_status]))
        metadata_status = pyogg.flac.FLAC__stream_decoder_process_until_end_of_metadata(self.decoder)
        if not metadata_status:
            raise DecodeException('An error occured when trying to decode the metadata of {}'.format(path))

    def read_callback(self, decoder, buffer, size, data):
        chunk = size.contents.value
        data = self.file.read(chunk)
        read_size = len(data)
        memmove(buffer, data, read_size)
        size.contents.value = read_size
        if read_size > 0:
            return 0
        elif read_size == 0:
            return 1
        else:
            return 2

    def seek_callback(self, decoder, offset, data):
        pos = self.file.seek(offset, 0)
        if pos < 0:
            return 1
        else:
            return 0

    def tell_callback(self, decoder, offset, data):
        """Decoder wants to know the current position of the file stream."""
        pos = self.file.tell()
        if pos < 0:
            return 1
        else:
            offset.contents.value = pos
            return 0

    def length_callback(self, decoder, length, data):
        """Decoder wants to know the total length of the stream."""
        if self.file_size == 0:
            return 1
        else:
            length.contents.value = self.file_size
            return 0

    def eof_callback(self, decoder, data):
        return self.file.tell() >= self.file_size