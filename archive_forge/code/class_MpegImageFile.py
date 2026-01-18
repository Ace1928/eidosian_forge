from __future__ import annotations
from . import Image, ImageFile
from ._binary import i8
class MpegImageFile(ImageFile.ImageFile):
    format = 'MPEG'
    format_description = 'MPEG'

    def _open(self):
        s = BitStream(self.fp)
        if s.read(32) != 435:
            msg = 'not an MPEG file'
            raise SyntaxError(msg)
        self._mode = 'RGB'
        self._size = (s.read(12), s.read(12))