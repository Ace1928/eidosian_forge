from __future__ import annotations
import gzip
import math
from . import Image, ImageFile
class FitsGzipDecoder(ImageFile.PyDecoder):
    _pulls_fd = True

    def decode(self, buffer):
        assert self.fd is not None
        value = gzip.decompress(self.fd.read())
        rows = []
        offset = 0
        number_of_bits = min(self.args[0] // 8, 4)
        for y in range(self.state.ysize):
            row = bytearray()
            for x in range(self.state.xsize):
                row += value[offset + (4 - number_of_bits):offset + 4]
                offset += 4
            rows.append(row)
        self.set_as_raw(bytes([pixel for row in rows[::-1] for pixel in row]))
        return (-1, 0)