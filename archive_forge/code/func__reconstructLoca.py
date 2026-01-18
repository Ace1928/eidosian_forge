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
def _reconstructLoca(self):
    """Return reconstructed loca table data."""
    if 'loca' not in self.ttFont:
        self.tables['glyf'].data = self.reconstructTable('glyf')
    locaTable = self.ttFont['loca']
    data = locaTable.compile(self.ttFont)
    if len(data) != self.tables['loca'].origLength:
        raise TTLibError("reconstructed 'loca' table doesn't match original size: expected %d, found %d" % (self.tables['loca'].origLength, len(data)))
    return data