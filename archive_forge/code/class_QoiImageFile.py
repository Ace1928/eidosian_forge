from __future__ import annotations
import os
from . import Image, ImageFile
from ._binary import i32be as i32
from ._binary import o8
class QoiImageFile(ImageFile.ImageFile):
    format = 'QOI'
    format_description = 'Quite OK Image'

    def _open(self):
        if not _accept(self.fp.read(4)):
            msg = 'not a QOI file'
            raise SyntaxError(msg)
        self._size = tuple((i32(self.fp.read(4)) for i in range(2)))
        channels = self.fp.read(1)[0]
        self._mode = 'RGB' if channels == 3 else 'RGBA'
        self.fp.seek(1, os.SEEK_CUR)
        self.tile = [('qoi', (0, 0) + self._size, self.fp.tell(), None)]