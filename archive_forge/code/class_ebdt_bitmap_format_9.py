from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
class ebdt_bitmap_format_9(BitmapPlusBigMetricsMixin, ComponentBitmapGlyph):

    def decompile(self):
        self.metrics = BigGlyphMetrics()
        dummy, data = sstruct.unpack2(bigGlyphMetricsFormat, self.data, self.metrics)
        numComponents, = struct.unpack('>H', data[:2])
        data = data[2:]
        self.componentArray = []
        for i in range(numComponents):
            curComponent = EbdtComponent()
            dummy, data = sstruct.unpack2(ebdtComponentFormat, data, curComponent)
            curComponent.name = self.ttFont.getGlyphName(curComponent.glyphCode)
            self.componentArray.append(curComponent)

    def compile(self, ttFont):
        dataList = []
        dataList.append(sstruct.pack(bigGlyphMetricsFormat, self.metrics))
        dataList.append(struct.pack('>H', len(self.componentArray)))
        for curComponent in self.componentArray:
            curComponent.glyphCode = ttFont.getGlyphID(curComponent.name)
            dataList.append(sstruct.pack(ebdtComponentFormat, curComponent))
        return bytesjoin(dataList)