from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class FixedSizeIndexSubTableMixin(object):

    def writeMetrics(self, writer, ttFont):
        writer.simpletag('imageSize', value=self.imageSize)
        writer.newline()
        self.metrics.toXML(writer, ttFont)

    def readMetrics(self, name, attrs, content, ttFont):
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name == 'imageSize':
                self.imageSize = safeEval(attrs['value'])
            elif name == BigGlyphMetrics.__name__:
                self.metrics = BigGlyphMetrics()
                self.metrics.fromXML(name, attrs, content, ttFont)
            elif name == SmallGlyphMetrics.__name__:
                log.warning('SmallGlyphMetrics being ignored in format %d.', self.indexFormat)

    def padBitmapData(self, data):
        assert len(data) <= self.imageSize, 'Data in indexSubTable format %d must be less than the fixed size.' % self.indexFormat
        pad = (self.imageSize - len(data)) * b'\x00'
        return data + pad