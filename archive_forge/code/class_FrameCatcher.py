import re
import sys
import time
import logging
import platform
import threading
import subprocess as sp
import imageio_ffmpeg
import numpy as np
from ..core import Format, image_as_uint
class FrameCatcher(threading.Thread):
    """Thread to keep reading the frame data from stdout. This is
    useful when streaming from a webcam. Otherwise, if the user code
    does not grab frames fast enough, the buffer will fill up, leading
    to lag, and ffmpeg can also stall (experienced on Linux). The
    get_frame() method always returns the last available image.
    """

    def __init__(self, gen):
        self._gen = gen
        self._frame = None
        self._frame_is_new = False
        self._lock = threading.RLock()
        threading.Thread.__init__(self)
        self.daemon = True
        self._should_stop = False
        self.start()

    def stop_me(self):
        self._should_stop = True
        while self.is_alive():
            time.sleep(0.001)

    def get_frame(self):
        while self._frame is None:
            time.sleep(0.001)
        with self._lock:
            is_new = self._frame_is_new
            self._frame_is_new = False
            return (self._frame, is_new)

    def run(self):
        try:
            while not self._should_stop:
                time.sleep(0)
                frame = self._gen.__next__()
                with self._lock:
                    self._frame = frame
                    self._frame_is_new = True
        except (StopIteration, EOFError):
            pass