from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
def _writeCodes(self, codes, writer):
    isUnicode = self.isUnicode()
    for code, name in codes:
        writer.simpletag('map', code=hex(code), name=name)
        if isUnicode:
            writer.comment(Unicode[code])
        writer.newline()