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
def _decodeCoordinates(self, glyph):
    data = self.nPointsStream
    endPtsOfContours = []
    endPoint = -1
    for i in range(glyph.numberOfContours):
        ptsOfContour, data = unpack255UShort(data)
        endPoint += ptsOfContour
        endPtsOfContours.append(endPoint)
    glyph.endPtsOfContours = endPtsOfContours
    self.nPointsStream = data
    self._decodeTriplets(glyph)
    self._decodeInstructions(glyph)