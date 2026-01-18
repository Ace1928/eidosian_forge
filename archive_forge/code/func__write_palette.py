from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
def _write_palette(self):
    data = b''
    palette = self.im.getpalette('RGBA', 'RGBA')
    for i in range(len(palette) // 4):
        r, g, b, a = palette[i * 4:(i + 1) * 4]
        data += struct.pack('<4B', b, g, r, a)
    while len(data) < 256 * 4:
        data += b'\x00' * 4
    return data