from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
class IndexCompiler(object):
    """Base class for writing CFF `INDEX data <https://docs.microsoft.com/en-us/typography/opentype/spec/cff2#5-index-data>`_
    to binary."""

    def __init__(self, items, strings, parent, isCFF2=None):
        if isCFF2 is None and hasattr(parent, 'isCFF2'):
            isCFF2 = parent.isCFF2
            assert isCFF2 is not None
        self.isCFF2 = isCFF2
        self.items = self.getItems(items, strings)
        self.parent = parent

    def getItems(self, items, strings):
        return items

    def getOffsets(self):
        if self.items:
            pos = 1
            offsets = [pos]
            for item in self.items:
                if hasattr(item, 'getDataLength'):
                    pos = pos + item.getDataLength()
                else:
                    pos = pos + len(item)
                offsets.append(pos)
        else:
            offsets = []
        return offsets

    def getDataLength(self):
        if self.isCFF2:
            countSize = 4
        else:
            countSize = 2
        if self.items:
            lastOffset = self.getOffsets()[-1]
            offSize = calcOffSize(lastOffset)
            dataLength = countSize + 1 + (len(self.items) + 1) * offSize + lastOffset - 1
        else:
            dataLength = countSize
        return dataLength

    def toFile(self, file):
        offsets = self.getOffsets()
        if self.isCFF2:
            writeCard32(file, len(self.items))
        else:
            writeCard16(file, len(self.items))
        if self.items:
            offSize = calcOffSize(offsets[-1])
            writeCard8(file, offSize)
            offSize = -offSize
            pack = struct.pack
            for offset in offsets:
                binOffset = pack('>l', offset)[offSize:]
                assert len(binOffset) == -offSize
                file.write(binOffset)
            for item in self.items:
                if hasattr(item, 'toFile'):
                    item.toFile(file)
                else:
                    data = tobytes(item, encoding='latin1')
                    file.write(data)