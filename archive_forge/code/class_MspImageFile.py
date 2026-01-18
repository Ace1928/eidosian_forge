from __future__ import annotations
import io
import struct
from . import Image, ImageFile
from ._binary import i16le as i16
from ._binary import o16le as o16
class MspImageFile(ImageFile.ImageFile):
    format = 'MSP'
    format_description = 'Windows Paint'

    def _open(self):
        s = self.fp.read(32)
        if not _accept(s):
            msg = 'not an MSP file'
            raise SyntaxError(msg)
        checksum = 0
        for i in range(0, 32, 2):
            checksum = checksum ^ i16(s, i)
        if checksum != 0:
            msg = 'bad MSP checksum'
            raise SyntaxError(msg)
        self._mode = '1'
        self._size = (i16(s, 4), i16(s, 6))
        if s[:4] == b'DanM':
            self.tile = [('raw', (0, 0) + self.size, 32, ('1', 0, 1))]
        else:
            self.tile = [('MSP', (0, 0) + self.size, 32, None)]