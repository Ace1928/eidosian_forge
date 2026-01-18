from __future__ import annotations
from . import Image, ImageFile
from ._binary import i16le as word
from ._binary import si16le as short
from ._binary import si32le as _long
class WmfHandler:

    def open(self, im):
        im._mode = 'RGB'
        self.bbox = im.info['wmf_bbox']

    def load(self, im):
        im.fp.seek(0)
        return Image.frombytes('RGB', im.size, Image.core.drawwmf(im.fp.read(), im.size, self.bbox), 'raw', 'BGR', im.size[0] * 3 + 3 & -4, -1)