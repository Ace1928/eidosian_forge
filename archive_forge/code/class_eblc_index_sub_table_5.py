from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class eblc_index_sub_table_5(FixedSizeIndexSubTableMixin, EblcIndexSubTable):

    def decompile(self):
        self.origDataLen = 0
        self.imageSize, = struct.unpack('>L', self.data[:4])
        data = self.data[4:]
        self.metrics, data = sstruct.unpack2(bigGlyphMetricsFormat, data, BigGlyphMetrics())
        numGlyphs, = struct.unpack('>L', data[:4])
        data = data[4:]
        glyphIds = [struct.unpack('>H', data[2 * i:2 * (i + 1)])[0] for i in range(numGlyphs)]
        offsets = [self.imageSize * i + self.imageDataOffset for i in range(len(glyphIds) + 1)]
        self.locations = list(zip(offsets, offsets[1:]))
        self.names = list(map(self.ttFont.getGlyphName, glyphIds))
        del self.data, self.ttFont

    def compile(self, ttFont):
        self.imageDataOffset = min(next(iter(zip(*self.locations))))
        dataList = [EblcIndexSubTable.compile(self, ttFont)]
        dataList.append(struct.pack('>L', self.imageSize))
        dataList.append(sstruct.pack(bigGlyphMetricsFormat, self.metrics))
        glyphIds = list(map(ttFont.getGlyphID, self.names))
        dataList.append(struct.pack('>L', len(glyphIds)))
        dataList += [struct.pack('>H', curId) for curId in glyphIds]
        if len(glyphIds) % 2 == 1:
            dataList.append(struct.pack('>H', 0))
        return bytesjoin(dataList)