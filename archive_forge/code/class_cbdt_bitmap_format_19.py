from fontTools.misc.textTools import bytesjoin
from fontTools.misc import sstruct
from . import E_B_D_T_
from .BitmapGlyphMetrics import (
from .E_B_D_T_ import (
import struct
class cbdt_bitmap_format_19(ColorBitmapGlyph):

    def decompile(self):
        dataLen, = struct.unpack('>L', self.data[:4])
        data = self.data[4:]
        assert dataLen <= len(data), 'Data overun in format 19'
        self.imageData = data[:dataLen]

    def compile(self, ttFont):
        return struct.pack('>L', len(self.imageData)) + self.imageData