from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def _calcFlavorDataOffsetsAndSize(self, start):
    """Calculate offsets and lengths for any meta- and/or private data."""
    offset = start
    data = self.flavorData
    if data.metaData:
        self.metaOrigLength = len(data.metaData)
        self.metaOffset = offset
        self.compressedMetaData = brotli.compress(data.metaData, mode=brotli.MODE_TEXT)
        self.metaLength = len(self.compressedMetaData)
        offset += self.metaLength
    else:
        self.metaOffset = self.metaLength = self.metaOrigLength = 0
        self.compressedMetaData = b''
    if data.privData:
        offset = offset + 3 & ~3
        self.privOffset = offset
        self.privLength = len(data.privData)
        offset += self.privLength
    else:
        self.privOffset = self.privLength = 0
    return offset