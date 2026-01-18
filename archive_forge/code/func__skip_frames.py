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
def _skip_frames(self, n=1):
    """Reads and throws away n frames"""
    for i in range(n):
        self._read_gen.__next__()
    self._pos += n