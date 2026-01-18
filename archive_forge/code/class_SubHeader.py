from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class SubHeader(object):

    def __init__(self):
        self.firstCode = None
        self.entryCount = None
        self.idDelta = None
        self.idRangeOffset = None
        self.glyphIndexArray = []