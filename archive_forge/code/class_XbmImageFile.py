from __future__ import annotations
import re
from . import Image, ImageFile
class XbmImageFile(ImageFile.ImageFile):
    format = 'XBM'
    format_description = 'X11 Bitmap'

    def _open(self):
        m = xbm_head.match(self.fp.read(512))
        if not m:
            msg = 'not a XBM file'
            raise SyntaxError(msg)
        xsize = int(m.group('width'))
        ysize = int(m.group('height'))
        if m.group('hotspot'):
            self.info['hotspot'] = (int(m.group('xhot')), int(m.group('yhot')))
        self._mode = '1'
        self._size = (xsize, ysize)
        self.tile = [('xbm', (0, 0) + self.size, m.end(), None)]