import gi
from gi.repository import GLib, Gst
import sys
import threading
import os
import queue
from urllib.parse import quote
from .exceptions import DecodeError
from .base import AudioFile
def _no_more_pads(self, element):
    """The callback for GstElement's "no-more-pads" signal.
        """
    if not self._got_a_pad:
        self.read_exc = NoStreamError()
        self.ready_sem.release()