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
def _decodeBBox(self, glyphID, glyph):
    haveBBox = bool(self.bboxBitmap[glyphID >> 3] & 128 >> (glyphID & 7))
    if glyph.isComposite() and (not haveBBox):
        raise TTLibError('no bbox values for composite glyph %d' % glyphID)
    if haveBBox:
        dummy, self.bboxStream = sstruct.unpack2(bboxFormat, self.bboxStream, glyph)
    else:
        glyph.recalcBounds(self)