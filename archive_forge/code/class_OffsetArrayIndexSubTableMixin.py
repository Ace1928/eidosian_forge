from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class OffsetArrayIndexSubTableMixin(object):

    def decompile(self):
        numGlyphs = self.lastGlyphIndex - self.firstGlyphIndex + 1
        indexingOffsets = [glyphIndex * offsetDataSize for glyphIndex in range(numGlyphs + 2)]
        indexingLocations = zip(indexingOffsets, indexingOffsets[1:])
        offsetArray = [struct.unpack(dataFormat, self.data[slice(*loc)])[0] for loc in indexingLocations]
        glyphIds = list(range(self.firstGlyphIndex, self.lastGlyphIndex + 1))
        modifiedOffsets = [offset + self.imageDataOffset for offset in offsetArray]
        self.locations = list(zip(modifiedOffsets, modifiedOffsets[1:]))
        self.names = list(map(self.ttFont.getGlyphName, glyphIds))
        self.removeSkipGlyphs()
        del self.data, self.ttFont

    def compile(self, ttFont):
        for curLoc, nxtLoc in zip(self.locations, self.locations[1:]):
            assert curLoc[1] == nxtLoc[0], 'Data must be consecutive in indexSubTable offset formats'
        glyphIds = list(map(ttFont.getGlyphID, self.names))
        assert all((glyphIds[i] < glyphIds[i + 1] for i in range(len(glyphIds) - 1)))
        idQueue = deque(glyphIds)
        locQueue = deque(self.locations)
        allGlyphIds = list(range(self.firstGlyphIndex, self.lastGlyphIndex + 1))
        allLocations = []
        for curId in allGlyphIds:
            if curId != idQueue[0]:
                allLocations.append((locQueue[0][0], locQueue[0][0]))
            else:
                idQueue.popleft()
                allLocations.append(locQueue.popleft())
        offsets = list(allLocations[0]) + [loc[1] for loc in allLocations[1:]]
        self.imageDataOffset = min(offsets)
        offsetArray = [offset - self.imageDataOffset for offset in offsets]
        dataList = [EblcIndexSubTable.compile(self, ttFont)]
        dataList += [struct.pack(dataFormat, offsetValue) for offsetValue in offsetArray]
        if offsetDataSize * len(offsetArray) % 4 != 0:
            dataList.append(struct.pack(dataFormat, 0))
        return bytesjoin(dataList)