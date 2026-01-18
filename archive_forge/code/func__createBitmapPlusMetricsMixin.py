from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def _createBitmapPlusMetricsMixin(metricsClass):
    metricStrings = [BigGlyphMetrics.__name__, SmallGlyphMetrics.__name__]
    curMetricsName = metricsClass.__name__
    metricsId = metricStrings.index(curMetricsName)
    oppositeMetricsName = metricStrings[1 - metricsId]

    class BitmapPlusMetricsMixin(object):

        def writeMetrics(self, writer, ttFont):
            self.metrics.toXML(writer, ttFont)

        def readMetrics(self, name, attrs, content, ttFont):
            for element in content:
                if not isinstance(element, tuple):
                    continue
                name, attrs, content = element
                if name == curMetricsName:
                    self.metrics = metricsClass()
                    self.metrics.fromXML(name, attrs, content, ttFont)
                elif name == oppositeMetricsName:
                    log.warning('Warning: %s being ignored in format %d.', oppositeMetricsName, self.getFormat())
    return BitmapPlusMetricsMixin