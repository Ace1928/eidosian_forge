import queue
import atexit
import weakref
import tempfile
from threading import Event, Thread
from pyglet.util import DecodeException
from .base import StreamingSource, AudioData, AudioFormat, StaticSource
from . import MediaEncoder, MediaDecoder
class _GLibMainLoopThread(Thread):
    """A background Thread for a GLib MainLoop"""

    def __init__(self):
        super().__init__(daemon=True)
        self.mainloop = GLib.MainLoop.new(None, False)
        self.start()

    def run(self):
        self.mainloop.run()