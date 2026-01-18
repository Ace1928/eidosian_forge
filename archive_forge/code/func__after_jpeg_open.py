from __future__ import annotations
import itertools
import os
import struct
from . import (
from ._binary import i16be as i16
from ._binary import o32le
def _after_jpeg_open(self, mpheader=None):
    self._initial_size = self.size
    self.mpinfo = mpheader if mpheader is not None else self._getmp()
    self.n_frames = self.mpinfo[45057]
    self.__mpoffsets = [mpent['DataOffset'] + self.info['mpoffset'] for mpent in self.mpinfo[45058]]
    self.__mpoffsets[0] = 0
    assert self.n_frames == len(self.__mpoffsets)
    del self.info['mpoffset']
    self.is_animated = self.n_frames > 1
    self._fp = self.fp
    self._fp.seek(self.__mpoffsets[0])
    self.__frame = 0
    self.offset = 0
    self.readonly = 1