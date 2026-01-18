from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class table_E_B_L_C_(DefaultTable.DefaultTable):
    dependencies = ['EBDT']

    def getIndexFormatClass(self, indexFormat):
        return eblc_sub_table_classes[indexFormat]

    def decompile(self, data, ttFont):
        origData = data
        i = 0
        dummy = sstruct.unpack(eblcHeaderFormat, data[:8], self)
        i += 8
        self.strikes = []
        for curStrikeIndex in range(self.numSizes):
            curStrike = Strike()
            self.strikes.append(curStrike)
            curTable = curStrike.bitmapSizeTable
            dummy = sstruct.unpack2(bitmapSizeTableFormatPart1, data[i:i + 16], curTable)
            i += 16
            for metric in ('hori', 'vert'):
                metricObj = SbitLineMetrics()
                vars(curTable)[metric] = metricObj
                dummy = sstruct.unpack2(sbitLineMetricsFormat, data[i:i + 12], metricObj)
                i += 12
            dummy = sstruct.unpack(bitmapSizeTableFormatPart2, data[i:i + 8], curTable)
            i += 8
        for curStrike in self.strikes:
            curTable = curStrike.bitmapSizeTable
            for subtableIndex in range(curTable.numberOfIndexSubTables):
                i = curTable.indexSubTableArrayOffset + subtableIndex * indexSubTableArraySize
                tup = struct.unpack(indexSubTableArrayFormat, data[i:i + indexSubTableArraySize])
                firstGlyphIndex, lastGlyphIndex, additionalOffsetToIndexSubtable = tup
                i = curTable.indexSubTableArrayOffset + additionalOffsetToIndexSubtable
                tup = struct.unpack(indexSubHeaderFormat, data[i:i + indexSubHeaderSize])
                indexFormat, imageFormat, imageDataOffset = tup
                indexFormatClass = self.getIndexFormatClass(indexFormat)
                indexSubTable = indexFormatClass(data[i + indexSubHeaderSize:], ttFont)
                indexSubTable.firstGlyphIndex = firstGlyphIndex
                indexSubTable.lastGlyphIndex = lastGlyphIndex
                indexSubTable.additionalOffsetToIndexSubtable = additionalOffsetToIndexSubtable
                indexSubTable.indexFormat = indexFormat
                indexSubTable.imageFormat = imageFormat
                indexSubTable.imageDataOffset = imageDataOffset
                indexSubTable.decompile()
                curStrike.indexSubTables.append(indexSubTable)

    def compile(self, ttFont):
        dataList = []
        self.numSizes = len(self.strikes)
        dataList.append(sstruct.pack(eblcHeaderFormat, self))
        dataSize = len(dataList[0])
        for _ in self.strikes:
            dataSize += sstruct.calcsize(bitmapSizeTableFormatPart1)
            dataSize += len(('hori', 'vert')) * sstruct.calcsize(sbitLineMetricsFormat)
            dataSize += sstruct.calcsize(bitmapSizeTableFormatPart2)
        indexSubTablePairDataList = []
        for curStrike in self.strikes:
            curTable = curStrike.bitmapSizeTable
            curTable.numberOfIndexSubTables = len(curStrike.indexSubTables)
            curTable.indexSubTableArrayOffset = dataSize
            sizeOfSubTableArray = curTable.numberOfIndexSubTables * indexSubTableArraySize
            lowerBound = dataSize
            dataSize += sizeOfSubTableArray
            upperBound = dataSize
            indexSubTableDataList = []
            for indexSubTable in curStrike.indexSubTables:
                indexSubTable.additionalOffsetToIndexSubtable = dataSize - curTable.indexSubTableArrayOffset
                glyphIds = list(map(ttFont.getGlyphID, indexSubTable.names))
                indexSubTable.firstGlyphIndex = min(glyphIds)
                indexSubTable.lastGlyphIndex = max(glyphIds)
                data = indexSubTable.compile(ttFont)
                indexSubTableDataList.append(data)
                dataSize += len(data)
            curTable.startGlyphIndex = min((ist.firstGlyphIndex for ist in curStrike.indexSubTables))
            curTable.endGlyphIndex = max((ist.lastGlyphIndex for ist in curStrike.indexSubTables))
            for i in curStrike.indexSubTables:
                data = struct.pack(indexSubHeaderFormat, i.firstGlyphIndex, i.lastGlyphIndex, i.additionalOffsetToIndexSubtable)
                indexSubTablePairDataList.append(data)
            indexSubTablePairDataList.extend(indexSubTableDataList)
            curTable.indexTablesSize = dataSize - curTable.indexSubTableArrayOffset
        for curStrike in self.strikes:
            curTable = curStrike.bitmapSizeTable
            data = sstruct.pack(bitmapSizeTableFormatPart1, curTable)
            dataList.append(data)
            for metric in ('hori', 'vert'):
                metricObj = vars(curTable)[metric]
                data = sstruct.pack(sbitLineMetricsFormat, metricObj)
                dataList.append(data)
            data = sstruct.pack(bitmapSizeTableFormatPart2, curTable)
            dataList.append(data)
        dataList.extend(indexSubTablePairDataList)
        return bytesjoin(dataList)

    def toXML(self, writer, ttFont):
        writer.simpletag('header', [('version', self.version)])
        writer.newline()
        for curIndex, curStrike in enumerate(self.strikes):
            curStrike.toXML(curIndex, writer, ttFont)

    def fromXML(self, name, attrs, content, ttFont):
        if name == 'header':
            self.version = safeEval(attrs['version'])
        elif name == 'strike':
            if not hasattr(self, 'strikes'):
                self.strikes = []
            strikeIndex = safeEval(attrs['index'])
            curStrike = Strike()
            curStrike.fromXML(name, attrs, content, ttFont, self)
            if strikeIndex >= len(self.strikes):
                self.strikes += [None] * (strikeIndex + 1 - len(self.strikes))
            assert self.strikes[strikeIndex] is None, 'Duplicate strike EBLC indices.'
            self.strikes[strikeIndex] = curStrike