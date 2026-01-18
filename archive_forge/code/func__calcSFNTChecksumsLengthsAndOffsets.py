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
def _calcSFNTChecksumsLengthsAndOffsets(self):
    """Compute the 'original' SFNT checksums, lengths and offsets for checksum
        adjustment calculation. Return the total size of the uncompressed font.
        """
    offset = sfntDirectorySize + sfntDirectoryEntrySize * len(self.tables)
    for tag, entry in self.tables.items():
        data = entry.data
        entry.origOffset = offset
        entry.origLength = len(data)
        if tag == 'head':
            entry.checkSum = calcChecksum(data[:8] + b'\x00\x00\x00\x00' + data[12:])
        else:
            entry.checkSum = calcChecksum(data)
        offset += entry.origLength + 3 & ~3
    return offset