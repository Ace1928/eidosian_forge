import gi
from gi.repository import GLib, Gst
import sys
import threading
import os
import queue
from urllib.parse import quote
from .exceptions import DecodeError
from .base import AudioFile
class UnknownTypeError(GStreamerError):
    """Raised when Gstreamer can't decode the given file type."""

    def __init__(self, streaminfo):
        super().__init__("can't decode stream: " + streaminfo)
        self.streaminfo = streaminfo