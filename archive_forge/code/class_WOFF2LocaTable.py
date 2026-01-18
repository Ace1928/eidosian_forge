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
class WOFF2LocaTable(getTableClass('loca')):
    """Same as parent class. The only difference is that it attempts to preserve
    the 'indexFormat' as encoded in the WOFF2 glyf table.
    """

    def __init__(self, tag=None):
        self.tableTag = Tag(tag or 'loca')

    def compile(self, ttFont):
        try:
            max_location = max(self.locations)
        except AttributeError:
            self.set([])
            max_location = 0
        if 'glyf' in ttFont and hasattr(ttFont['glyf'], 'indexFormat'):
            indexFormat = ttFont['glyf'].indexFormat
            if indexFormat == 0:
                if max_location >= 131072:
                    raise TTLibError('indexFormat is 0 but local offsets > 0x20000')
                if not all((l % 2 == 0 for l in self.locations)):
                    raise TTLibError('indexFormat is 0 but local offsets not multiples of 2')
                locations = array.array('H')
                for i in range(len(self.locations)):
                    locations.append(self.locations[i] // 2)
            else:
                locations = array.array('I', self.locations)
            if sys.byteorder != 'big':
                locations.byteswap()
            data = locations.tobytes()
        else:
            data = super(WOFF2LocaTable, self).compile(ttFont)
        return data