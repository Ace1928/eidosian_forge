from __future__ import annotations
import io
import os
import struct
from . import Image, ImageFile, _binary
def _parse_comment(self):
    hdr = self.fp.read(2)
    length = _binary.i16be(hdr)
    self.fp.seek(length - 2, os.SEEK_CUR)
    while True:
        marker = self.fp.read(2)
        if not marker:
            break
        typ = marker[1]
        if typ in (144, 217):
            break
        hdr = self.fp.read(2)
        length = _binary.i16be(hdr)
        if typ == 100:
            self.info['comment'] = self.fp.read(length - 2)[2:]
            break
        else:
            self.fp.seek(length - 2, os.SEEK_CUR)