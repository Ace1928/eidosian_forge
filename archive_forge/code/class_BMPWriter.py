import os
import os.path
import struct
from io import BytesIO
from typing import BinaryIO, Tuple
from .jbig2 import JBIG2StreamReader, JBIG2StreamWriter
from .layout import LTImage
from .pdfcolor import LITERAL_DEVICE_CMYK
from .pdfcolor import LITERAL_DEVICE_GRAY
from .pdfcolor import LITERAL_DEVICE_RGB
from .pdftypes import (
class BMPWriter:

    def __init__(self, fp: BinaryIO, bits: int, width: int, height: int) -> None:
        self.fp = fp
        self.bits = bits
        self.width = width
        self.height = height
        if bits == 1:
            ncols = 2
        elif bits == 8:
            ncols = 256
        elif bits == 24:
            ncols = 0
        else:
            raise ValueError(bits)
        self.linesize = align32((self.width * self.bits + 7) // 8)
        self.datasize = self.linesize * self.height
        headersize = 14 + 40 + ncols * 4
        info = struct.pack('<IiiHHIIIIII', 40, self.width, self.height, 1, self.bits, 0, self.datasize, 0, 0, ncols, 0)
        assert len(info) == 40, str(len(info))
        header = struct.pack('<ccIHHI', b'B', b'M', headersize + self.datasize, 0, 0, headersize)
        assert len(header) == 14, str(len(header))
        self.fp.write(header)
        self.fp.write(info)
        if ncols == 2:
            for i in (0, 255):
                self.fp.write(struct.pack('BBBx', i, i, i))
        elif ncols == 256:
            for i in range(256):
                self.fp.write(struct.pack('BBBx', i, i, i))
        self.pos0 = self.fp.tell()
        self.pos1 = self.pos0 + self.datasize

    def write_line(self, y: int, data: bytes) -> None:
        self.fp.seek(self.pos1 - (y + 1) * self.linesize)
        self.fp.write(data)