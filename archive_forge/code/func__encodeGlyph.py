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
def _encodeGlyph(self, glyphID):
    glyphName = self.getGlyphName(glyphID)
    glyph = self[glyphName]
    self.nContourStream += struct.pack('>h', glyph.numberOfContours)
    if glyph.numberOfContours == 0:
        return
    elif glyph.isComposite():
        self._encodeComponents(glyph)
    elif glyph.isVarComposite():
        raise NotImplementedError
    else:
        self._encodeCoordinates(glyph)
        self._encodeOverlapSimpleFlag(glyph, glyphID)
    self._encodeBBox(glyphID, glyph)