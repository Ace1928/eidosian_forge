from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
class BLPEncoder(ImageFile.PyEncoder):
    _pushes_fd = True

    def _write_palette(self):
        data = b''
        palette = self.im.getpalette('RGBA', 'RGBA')
        for i in range(len(palette) // 4):
            r, g, b, a = palette[i * 4:(i + 1) * 4]
            data += struct.pack('<4B', b, g, r, a)
        while len(data) < 256 * 4:
            data += b'\x00' * 4
        return data

    def encode(self, bufsize):
        palette_data = self._write_palette()
        offset = 20 + 16 * 4 * 2 + len(palette_data)
        data = struct.pack('<16I', offset, *(0,) * 15)
        w, h = self.im.size
        data += struct.pack('<16I', w * h, *(0,) * 15)
        data += palette_data
        for y in range(h):
            for x in range(w):
                data += struct.pack('<B', self.im.getpixel((x, y)))
        return (len(data), 0, data)