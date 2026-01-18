from __future__ import annotations
import os
import struct
import sys
from . import Image, ImageFile
class SpiderImageFile(ImageFile.ImageFile):
    format = 'SPIDER'
    format_description = 'Spider 2D image'
    _close_exclusive_fp_after_loading = False

    def _open(self):
        n = 27 * 4
        f = self.fp.read(n)
        try:
            self.bigendian = 1
            t = struct.unpack('>27f', f)
            hdrlen = isSpiderHeader(t)
            if hdrlen == 0:
                self.bigendian = 0
                t = struct.unpack('<27f', f)
                hdrlen = isSpiderHeader(t)
            if hdrlen == 0:
                msg = 'not a valid Spider file'
                raise SyntaxError(msg)
        except struct.error as e:
            msg = 'not a valid Spider file'
            raise SyntaxError(msg) from e
        h = (99,) + t
        iform = int(h[5])
        if iform != 1:
            msg = 'not a Spider 2D image'
            raise SyntaxError(msg)
        self._size = (int(h[12]), int(h[2]))
        self.istack = int(h[24])
        self.imgnumber = int(h[27])
        if self.istack == 0 and self.imgnumber == 0:
            offset = hdrlen
            self._nimages = 1
        elif self.istack > 0 and self.imgnumber == 0:
            self.imgbytes = int(h[12]) * int(h[2]) * 4
            self.hdrlen = hdrlen
            self._nimages = int(h[26])
            offset = hdrlen * 2
            self.imgnumber = 1
        elif self.istack == 0 and self.imgnumber > 0:
            offset = hdrlen + self.stkoffset
            self.istack = 2
        else:
            msg = 'inconsistent stack header values'
            raise SyntaxError(msg)
        if self.bigendian:
            self.rawmode = 'F;32BF'
        else:
            self.rawmode = 'F;32F'
        self._mode = 'F'
        self.tile = [('raw', (0, 0) + self.size, offset, (self.rawmode, 0, 1))]
        self._fp = self.fp

    @property
    def n_frames(self):
        return self._nimages

    @property
    def is_animated(self):
        return self._nimages > 1

    def tell(self):
        if self.imgnumber < 1:
            return 0
        else:
            return self.imgnumber - 1

    def seek(self, frame):
        if self.istack == 0:
            msg = 'attempt to seek in a non-stack file'
            raise EOFError(msg)
        if not self._seek_check(frame):
            return
        self.stkoffset = self.hdrlen + frame * (self.hdrlen + self.imgbytes)
        self.fp = self._fp
        self.fp.seek(self.stkoffset)
        self._open()

    def convert2byte(self, depth=255):
        minimum, maximum = self.getextrema()
        m = 1
        if maximum != minimum:
            m = depth / (maximum - minimum)
        b = -m * minimum
        return self.point(lambda i, m=m, b=b: i * m + b).convert('L')

    def tkPhotoImage(self):
        from . import ImageTk
        return ImageTk.PhotoImage(self.convert2byte(), palette=256)