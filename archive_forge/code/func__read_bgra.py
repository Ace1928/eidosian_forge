from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
def _read_bgra(self, palette):
    data = bytearray()
    _data = BytesIO(self._safe_read(self._blp_lengths[0]))
    while True:
        try:
            offset, = struct.unpack('<B', _data.read(1))
        except struct.error:
            break
        b, g, r, a = palette[offset]
        d = (r, g, b)
        if self._blp_alpha_depth:
            d += (a,)
        data.extend(d)
    return data