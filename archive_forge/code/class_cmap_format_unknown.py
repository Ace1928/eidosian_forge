from fontTools.misc.textTools import bytesjoin, safeEval, readHex
from fontTools.misc.encodingTools import getEncoding
from fontTools.ttLib import getSearchRange
from fontTools.unicode import Unicode
from . import DefaultTable
import sys
import struct
import array
import logging
class cmap_format_unknown(CmapSubtable):

    def toXML(self, writer, ttFont):
        cmapName = self.__class__.__name__[:12] + str(self.format)
        writer.begintag(cmapName, [('platformID', self.platformID), ('platEncID', self.platEncID)])
        writer.newline()
        writer.dumphex(self.data)
        writer.endtag(cmapName)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.data = readHex(content)
        self.cmap = {}

    def decompileHeader(self, data, ttFont):
        self.language = 0
        self.data = data

    def decompile(self, data, ttFont):
        if data is not None and ttFont is not None:
            self.decompileHeader(data, ttFont)
        else:
            assert data is None and ttFont is None, 'Need both data and ttFont arguments'

    def compile(self, ttFont):
        if self.data:
            return self.data
        else:
            return None