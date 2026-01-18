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
class CFFWriter(object):
    """Helper class for serializing CFF data to binary. Used by
    :meth:`CFFFontSet.compile`."""

    def __init__(self, isCFF2):
        self.data = []
        self.isCFF2 = isCFF2

    def add(self, table):
        self.data.append(table)

    def toFile(self, file):
        lastPosList = None
        count = 1
        while True:
            log.log(DEBUG, 'CFFWriter.toFile() iteration: %d', count)
            count = count + 1
            pos = 0
            posList = [pos]
            for item in self.data:
                if hasattr(item, 'getDataLength'):
                    endPos = pos + item.getDataLength()
                    if isinstance(item, TopDictIndexCompiler) and item.isCFF2:
                        self.topDictSize = item.getDataLength()
                else:
                    endPos = pos + len(item)
                if hasattr(item, 'setPos'):
                    item.setPos(pos, endPos)
                pos = endPos
                posList.append(pos)
            if posList == lastPosList:
                break
            lastPosList = posList
        log.log(DEBUG, 'CFFWriter.toFile() writing to file.')
        begin = file.tell()
        if self.isCFF2:
            self.data[1] = struct.pack('>H', self.topDictSize)
        else:
            self.offSize = calcOffSize(lastPosList[-1])
            self.data[1] = struct.pack('B', self.offSize)
        posList = [0]
        for item in self.data:
            if hasattr(item, 'toFile'):
                item.toFile(file)
            else:
                file.write(item)
            posList.append(file.tell() - begin)
        assert posList == lastPosList