import os
import time
import warnings
import numpy as np
from .. import _HAS_FFMPEG
from ..utils import *
def _read_frame_data(self):
    framesize = self.outputdepth * self.outputwidth * self.outputheight
    assert self._proc is not None
    try:
        arr = np.frombuffer(self._proc.stdout.read(framesize * self.dtype.itemsize), dtype=self.dtype)
        if len(arr) != framesize:
            return np.array([])
    except Exception as err:
        self._terminate()
        err1 = str(err)
        raise RuntimeError('%s' % (err1,))
    return arr