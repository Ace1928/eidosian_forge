from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class GlyphID(SimpleValue):
    staticSize = 2
    typecode = 'H'

    def readArray(self, reader, font, tableDict, count):
        return font.getGlyphNameMany(reader.readArray(self.typecode, self.staticSize, count))

    def read(self, reader, font, tableDict):
        return font.getGlyphName(reader.readValue(self.typecode, self.staticSize))

    def writeArray(self, writer, font, tableDict, values):
        writer.writeArray(self.typecode, font.getGlyphIDMany(values))

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeValue(self.typecode, font.getGlyphID(value))