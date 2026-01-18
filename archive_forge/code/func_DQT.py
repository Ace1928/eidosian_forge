from __future__ import annotations
import array
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import warnings
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from .JpegPresets import presets
def DQT(self, marker):
    n = i16(self.fp.read(2)) - 2
    s = ImageFile._safe_read(self.fp, n)
    while len(s):
        v = s[0]
        precision = 1 if v // 16 == 0 else 2
        qt_length = 1 + precision * 64
        if len(s) < qt_length:
            msg = 'bad quantization table marker'
            raise SyntaxError(msg)
        data = array.array('B' if precision == 1 else 'H', s[1:qt_length])
        if sys.byteorder == 'little' and precision > 1:
            data.byteswap()
        self.quantization[v & 15] = [data[i] for i in zigzag_index]
        s = s[qt_length:]