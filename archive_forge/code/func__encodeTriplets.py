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
def _encodeTriplets(self, glyph):
    assert len(glyph.coordinates) == len(glyph.flags)
    coordinates = glyph.coordinates.copy()
    coordinates.absoluteToRelative()
    flags = array.array('B')
    triplets = array.array('B')
    for i in range(len(coordinates)):
        onCurve = glyph.flags[i] & _g_l_y_f.flagOnCurve
        x, y = coordinates[i]
        absX = abs(x)
        absY = abs(y)
        onCurveBit = 0 if onCurve else 128
        xSignBit = 0 if x < 0 else 1
        ySignBit = 0 if y < 0 else 1
        xySignBits = xSignBit + 2 * ySignBit
        if x == 0 and absY < 1280:
            flags.append(onCurveBit + ((absY & 3840) >> 7) + ySignBit)
            triplets.append(absY & 255)
        elif y == 0 and absX < 1280:
            flags.append(onCurveBit + 10 + ((absX & 3840) >> 7) + xSignBit)
            triplets.append(absX & 255)
        elif absX < 65 and absY < 65:
            flags.append(onCurveBit + 20 + (absX - 1 & 48) + ((absY - 1 & 48) >> 2) + xySignBits)
            triplets.append((absX - 1 & 15) << 4 | absY - 1 & 15)
        elif absX < 769 and absY < 769:
            flags.append(onCurveBit + 84 + 12 * ((absX - 1 & 768) >> 8) + ((absY - 1 & 768) >> 6) + xySignBits)
            triplets.append(absX - 1 & 255)
            triplets.append(absY - 1 & 255)
        elif absX < 4096 and absY < 4096:
            flags.append(onCurveBit + 120 + xySignBits)
            triplets.append(absX >> 4)
            triplets.append((absX & 15) << 4 | absY >> 8)
            triplets.append(absY & 255)
        else:
            flags.append(onCurveBit + 124 + xySignBits)
            triplets.append(absX >> 8)
            triplets.append(absX & 255)
            triplets.append(absY >> 8)
            triplets.append(absY & 255)
    self.flagStream += flags.tobytes()
    self.glyphStream += triplets.tobytes()