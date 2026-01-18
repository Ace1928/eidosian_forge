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
def _calcTotalSize(self):
    """Calculate total size of WOFF2 font, including any meta- and/or private data."""
    offset = self.directorySize
    for entry in self.tables.values():
        offset += len(entry.toString())
    offset += self.totalCompressedSize
    offset = offset + 3 & ~3
    offset = self._calcFlavorDataOffsetsAndSize(offset)
    return offset