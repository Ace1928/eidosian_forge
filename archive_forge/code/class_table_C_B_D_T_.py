from fontTools.misc.textTools import bytesjoin
from fontTools.misc import sstruct
from . import E_B_D_T_
from .BitmapGlyphMetrics import (
from .E_B_D_T_ import (
import struct
class table_C_B_D_T_(E_B_D_T_.table_E_B_D_T_):
    locatorName = 'CBLC'

    def getImageFormatClass(self, imageFormat):
        try:
            return E_B_D_T_.table_E_B_D_T_.getImageFormatClass(self, imageFormat)
        except KeyError:
            return cbdt_bitmap_classes[imageFormat]