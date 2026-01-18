from __future__ import annotations
import warnings
from . import Image, ImageFile, ImagePalette
from ._binary import i16le as i16
from ._binary import o8
from ._binary import o16le as o16
class TgaImageFile(ImageFile.ImageFile):
    format = 'TGA'
    format_description = 'Targa'

    def _open(self):
        s = self.fp.read(18)
        id_len = s[0]
        colormaptype = s[1]
        imagetype = s[2]
        depth = s[16]
        flags = s[17]
        self._size = (i16(s, 12), i16(s, 14))
        if colormaptype not in (0, 1) or self.size[0] <= 0 or self.size[1] <= 0 or (depth not in (1, 8, 16, 24, 32)):
            msg = 'not a TGA file'
            raise SyntaxError(msg)
        if imagetype in (3, 11):
            self._mode = 'L'
            if depth == 1:
                self._mode = '1'
            elif depth == 16:
                self._mode = 'LA'
        elif imagetype in (1, 9):
            self._mode = 'P'
        elif imagetype in (2, 10):
            self._mode = 'RGB'
            if depth == 32:
                self._mode = 'RGBA'
        else:
            msg = 'unknown TGA mode'
            raise SyntaxError(msg)
        orientation = flags & 48
        self._flip_horizontally = orientation in [16, 48]
        if orientation in [32, 48]:
            orientation = 1
        elif orientation in [0, 16]:
            orientation = -1
        else:
            msg = 'unknown TGA orientation'
            raise SyntaxError(msg)
        self.info['orientation'] = orientation
        if imagetype & 8:
            self.info['compression'] = 'tga_rle'
        if id_len:
            self.info['id_section'] = self.fp.read(id_len)
        if colormaptype:
            start, size, mapdepth = (i16(s, 3), i16(s, 5), s[7])
            if mapdepth == 16:
                self.palette = ImagePalette.raw('BGR;15', b'\x00' * 2 * start + self.fp.read(2 * size))
            elif mapdepth == 24:
                self.palette = ImagePalette.raw('BGR', b'\x00' * 3 * start + self.fp.read(3 * size))
            elif mapdepth == 32:
                self.palette = ImagePalette.raw('BGRA', b'\x00' * 4 * start + self.fp.read(4 * size))
        try:
            rawmode = MODES[imagetype & 7, depth]
            if imagetype & 8:
                self.tile = [('tga_rle', (0, 0) + self.size, self.fp.tell(), (rawmode, orientation, depth))]
            else:
                self.tile = [('raw', (0, 0) + self.size, self.fp.tell(), (rawmode, 0, orientation))]
        except KeyError:
            pass

    def load_end(self):
        if self._flip_horizontally:
            self.im = self.im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)