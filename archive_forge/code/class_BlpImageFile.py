from __future__ import annotations
import os
import struct
from enum import IntEnum
from io import BytesIO
from . import Image, ImageFile
class BlpImageFile(ImageFile.ImageFile):
    """
    Blizzard Mipmap Format
    """
    format = 'BLP'
    format_description = 'Blizzard Mipmap Format'

    def _open(self):
        self.magic = self.fp.read(4)
        self.fp.seek(5, os.SEEK_CUR)
        self._blp_alpha_depth, = struct.unpack('<b', self.fp.read(1))
        self.fp.seek(2, os.SEEK_CUR)
        self._size = struct.unpack('<II', self.fp.read(8))
        if self.magic in (b'BLP1', b'BLP2'):
            decoder = self.magic.decode()
        else:
            msg = f'Bad BLP magic {repr(self.magic)}'
            raise BLPFormatError(msg)
        self._mode = 'RGBA' if self._blp_alpha_depth else 'RGB'
        self.tile = [(decoder, (0, 0) + self.size, 0, (self.mode, 0, 1))]