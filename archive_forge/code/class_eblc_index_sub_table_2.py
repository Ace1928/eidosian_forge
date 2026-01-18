from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class eblc_index_sub_table_2(FixedSizeIndexSubTableMixin, EblcIndexSubTable):

    def decompile(self):
        self.imageSize, = struct.unpack('>L', self.data[:4])
        self.metrics = BigGlyphMetrics()
        sstruct.unpack2(bigGlyphMetricsFormat, self.data[4:], self.metrics)
        glyphIds = list(range(self.firstGlyphIndex, self.lastGlyphIndex + 1))
        offsets = [self.imageSize * i + self.imageDataOffset for i in range(len(glyphIds) + 1)]
        self.locations = list(zip(offsets, offsets[1:]))
        self.names = list(map(self.ttFont.getGlyphName, glyphIds))
        del self.data, self.ttFont

    def compile(self, ttFont):
        glyphIds = list(map(ttFont.getGlyphID, self.names))
        assert glyphIds == list(range(self.firstGlyphIndex, self.lastGlyphIndex + 1)), 'Format 2 ids must be consecutive.'
        self.imageDataOffset = min(next(iter(zip(*self.locations))))
        dataList = [EblcIndexSubTable.compile(self, ttFont)]
        dataList.append(struct.pack('>L', self.imageSize))
        dataList.append(sstruct.pack(bigGlyphMetricsFormat, self.metrics))
        return bytesjoin(dataList)