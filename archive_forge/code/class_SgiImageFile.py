from __future__ import annotations
import os
import struct
from . import Image, ImageFile
from ._binary import i16be as i16
from ._binary import o8
class SgiImageFile(ImageFile.ImageFile):
    format = 'SGI'
    format_description = 'SGI Image File Format'

    def _open(self):
        headlen = 512
        s = self.fp.read(headlen)
        if not _accept(s):
            msg = 'Not an SGI image file'
            raise ValueError(msg)
        compression = s[2]
        bpc = s[3]
        dimension = i16(s, 4)
        xsize = i16(s, 6)
        ysize = i16(s, 8)
        zsize = i16(s, 10)
        layout = (bpc, dimension, zsize)
        rawmode = ''
        try:
            rawmode = MODES[layout]
        except KeyError:
            pass
        if rawmode == '':
            msg = 'Unsupported SGI image mode'
            raise ValueError(msg)
        self._size = (xsize, ysize)
        self._mode = rawmode.split(';')[0]
        if self.mode == 'RGB':
            self.custom_mimetype = 'image/rgb'
        orientation = -1
        if compression == 0:
            pagesize = xsize * ysize * bpc
            if bpc == 2:
                self.tile = [('SGI16', (0, 0) + self.size, headlen, (self.mode, 0, orientation))]
            else:
                self.tile = []
                offset = headlen
                for layer in self.mode:
                    self.tile.append(('raw', (0, 0) + self.size, offset, (layer, 0, orientation)))
                    offset += pagesize
        elif compression == 1:
            self.tile = [('sgi_rle', (0, 0) + self.size, headlen, (rawmode, orientation, bpc))]