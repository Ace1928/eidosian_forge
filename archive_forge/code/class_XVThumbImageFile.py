from __future__ import annotations
from . import Image, ImageFile, ImagePalette
from ._binary import o8
class XVThumbImageFile(ImageFile.ImageFile):
    format = 'XVThumb'
    format_description = 'XV thumbnail image'

    def _open(self):
        if not _accept(self.fp.read(6)):
            msg = 'not an XV thumbnail file'
            raise SyntaxError(msg)
        self.fp.readline()
        while True:
            s = self.fp.readline()
            if not s:
                msg = 'Unexpected EOF reading XV thumbnail file'
                raise SyntaxError(msg)
            if s[0] != 35:
                break
        s = s.strip().split()
        self._mode = 'P'
        self._size = (int(s[0]), int(s[1]))
        self.palette = ImagePalette.raw('RGB', PALETTE)
        self.tile = [('raw', (0, 0) + self.size, self.fp.tell(), (self.mode, 0, 1))]