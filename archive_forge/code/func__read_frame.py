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
def _read_frame(self):
    w, h = self._meta['size']
    framesize = w * h * self._depth * self._bytes_per_channel
    if self._frame_catcher:
        s, is_new = self._frame_catcher.get_frame()
    else:
        s = self._read_gen.__next__()
        is_new = True
    if len(s) != framesize:
        raise RuntimeError('Frame is %i bytes, but expected %i.' % (len(s), framesize))
    result = np.frombuffer(s, dtype=self._dtype).copy()
    result = result.reshape((h, w, self._depth))
    self._lastread = result
    return (result, is_new)