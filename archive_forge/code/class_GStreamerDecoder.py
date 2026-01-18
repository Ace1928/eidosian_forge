import queue
import atexit
import weakref
import tempfile
from threading import Event, Thread
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
class GStreamerDecoder(MediaDecoder):

    def __init__(self):
        Gst.init(None)
        self._glib_loop = _GLibMainLoopThread()

    def get_file_extensions(self):
        return ('.mp3', '.flac', '.ogg', '.m4a')

    def decode(self, filename, file, streaming=True):
        if not any((filename.endswith(ext) for ext in self.get_file_extensions())):
            raise GStreamerDecodeException('Unsupported format.')
        if streaming:
            return GStreamerSource(filename, file)
        else:
            return StaticSource(GStreamerSource(filename, file))