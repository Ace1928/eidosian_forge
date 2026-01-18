from ctypes import memmove, byref, c_uint32, sizeof, cast, c_void_p, create_string_buffer, POINTER, c_char, \
from pyglet.libs.darwin import cf, CFSTR
from pyglet.libs.darwin.coreaudio import kCFURLPOSIXPathStyle, AudioStreamBasicDescription, ca, ExtAudioFileRef, \
from pyglet.media import StreamingSource, StaticSource
from pyglet.media.codecs import AudioFormat, MediaDecoder, AudioData
class MemoryFileObject:

    def __init__(self, file):
        self.file = file
        if not getattr(self.file, 'seek', None) or not getattr(self.file, 'tell', None):
            raise Exception('File object does not support seeking.')
        self.file.seek(0, 2)
        self.file_size = self.file.tell()
        self.file.seek(0)
        self.data = []

        def read_data_cb(ref, offset, requested_length, buffer, actual_count):
            self.file.seek(offset)
            data = self.file.read(requested_length)
            data_size = len(data)
            memmove(buffer, data, data_size)
            actual_count.contents.value = data_size
            return 0

        def getsize_cb(ref):
            return self.file_size
        self.getsize_func = AudioFile_GetSizeProc(getsize_cb)
        self.read_func = AudioFile_ReadProc(read_data_cb)