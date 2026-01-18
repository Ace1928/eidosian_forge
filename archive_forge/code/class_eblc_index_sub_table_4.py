from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class eblc_index_sub_table_4(EblcIndexSubTable):

    def decompile(self):
        numGlyphs, = struct.unpack('>L', self.data[:4])
        data = self.data[4:]
        indexingOffsets = [glyphIndex * codeOffsetPairSize for glyphIndex in range(numGlyphs + 2)]
        indexingLocations = zip(indexingOffsets, indexingOffsets[1:])
        glyphArray = [struct.unpack(codeOffsetPairFormat, data[slice(*loc)]) for loc in indexingLocations]
        glyphIds, offsets = list(map(list, zip(*glyphArray)))
        glyphIds.pop()
        offsets = [offset + self.imageDataOffset for offset in offsets]
        self.locations = list(zip(offsets, offsets[1:]))
        self.names = list(map(self.ttFont.getGlyphName, glyphIds))
        del self.data, self.ttFont

    def compile(self, ttFont):
        for curLoc, nxtLoc in zip(self.locations, self.locations[1:]):
            assert curLoc[1] == nxtLoc[0], 'Data must be consecutive in indexSubTable format 4'
        offsets = list(self.locations[0]) + [loc[1] for loc in self.locations[1:]]
        self.imageDataOffset = min(offsets)
        offsets = [offset - self.imageDataOffset for offset in offsets]
        glyphIds = list(map(ttFont.getGlyphID, self.names))
        idsPlusPad = list(itertools.chain(glyphIds, [0]))
        dataList = [EblcIndexSubTable.compile(self, ttFont)]
        dataList.append(struct.pack('>L', len(glyphIds)))
        tmp = [struct.pack(codeOffsetPairFormat, *cop) for cop in zip(idsPlusPad, offsets)]
        dataList += tmp
        data = bytesjoin(dataList)
        return data