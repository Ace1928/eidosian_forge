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
def _decodeOverlapSimpleFlag(self, glyph, glyphID):
    if self.overlapSimpleBitmap is None or glyph.numberOfContours <= 0:
        return
    byte = glyphID >> 3
    bit = glyphID & 7
    if self.overlapSimpleBitmap[byte] & 128 >> bit:
        glyph.flags[0] |= _g_l_y_f.flagOverlapSimple