from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class BitmapSizeTable(object):

    def _getXMLMetricNames(self):
        dataNames = sstruct.getformat(bitmapSizeTableFormatPart1)[1]
        dataNames = dataNames + sstruct.getformat(bitmapSizeTableFormatPart2)[1]
        return dataNames[3:]

    def toXML(self, writer, ttFont):
        writer.begintag('bitmapSizeTable')
        writer.newline()
        for metric in ('hori', 'vert'):
            getattr(self, metric).toXML(metric, writer, ttFont)
        for metricName in self._getXMLMetricNames():
            writer.simpletag(metricName, value=getattr(self, metricName))
            writer.newline()
        writer.endtag('bitmapSizeTable')
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        dataNames = set(self._getXMLMetricNames())
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name == 'sbitLineMetrics':
                direction = attrs['direction']
                assert direction in ('hori', 'vert'), 'SbitLineMetrics direction specified invalid.'
                metricObj = SbitLineMetrics()
                metricObj.fromXML(name, attrs, content, ttFont)
                vars(self)[direction] = metricObj
            elif name in dataNames:
                vars(self)[name] = safeEval(attrs['value'])
            else:
                log.warning("unknown name '%s' being ignored in BitmapSizeTable.", name)