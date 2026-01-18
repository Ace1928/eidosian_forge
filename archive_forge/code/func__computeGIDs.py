from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
def _computeGIDs(self, startingGlyph, numberOfGlyphs):
    return [startingGlyph] * numberOfGlyphs