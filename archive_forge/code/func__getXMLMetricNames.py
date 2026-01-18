from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
def _getXMLMetricNames(self):
    dataNames = sstruct.getformat(bitmapSizeTableFormatPart1)[1]
    dataNames = dataNames + sstruct.getformat(bitmapSizeTableFormatPart2)[1]
    return dataNames[3:]