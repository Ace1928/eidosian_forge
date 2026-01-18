from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def _encodeCoordinates(self, glyph):
    lastEndPoint = -1
    if _g_l_y_f.flagCubic in glyph.flags:
        raise NotImplementedError
    for endPoint in glyph.endPtsOfContours:
        ptsOfContour = endPoint - lastEndPoint
        self.nPointsStream += pack255UShort(ptsOfContour)
        lastEndPoint = endPoint
    self._encodeTriplets(glyph)
    self._encodeInstructions(glyph)