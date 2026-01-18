from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
class EblcIndexSubTable(object):

    def __init__(self, data, ttFont):
        self.data = data
        self.ttFont = ttFont

    def __getattr__(self, attr):
        if attr[:2] == '__':
            raise AttributeError(attr)
        if attr == 'data':
            raise AttributeError(attr)
        self.decompile()
        return getattr(self, attr)

    def ensureDecompiled(self, recurse=False):
        if hasattr(self, 'data'):
            self.decompile()

    def compile(self, ttFont):
        return struct.pack(indexSubHeaderFormat, self.indexFormat, self.imageFormat, self.imageDataOffset)

    def toXML(self, writer, ttFont):
        writer.begintag(self.__class__.__name__, [('imageFormat', self.imageFormat), ('firstGlyphIndex', self.firstGlyphIndex), ('lastGlyphIndex', self.lastGlyphIndex)])
        writer.newline()
        self.writeMetrics(writer, ttFont)
        glyphIds = map(ttFont.getGlyphID, self.names)
        for glyphName, glyphId in zip(self.names, glyphIds):
            writer.simpletag('glyphLoc', name=glyphName, id=glyphId)
            writer.newline()
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.imageFormat = safeEval(attrs['imageFormat'])
        self.firstGlyphIndex = safeEval(attrs['firstGlyphIndex'])
        self.lastGlyphIndex = safeEval(attrs['lastGlyphIndex'])
        self.readMetrics(name, attrs, content, ttFont)
        self.names = []
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attrs, content = element
            if name == 'glyphLoc':
                self.names.append(attrs['name'])

    def writeMetrics(self, writer, ttFont):
        pass

    def readMetrics(self, name, attrs, content, ttFont):
        pass

    def padBitmapData(self, data):
        return data

    def removeSkipGlyphs(self):

        def isValidLocation(args):
            name, (startByte, endByte) = args
            return startByte < endByte
        dataPairs = list(filter(isValidLocation, zip(self.names, self.locations)))
        self.names, self.locations = list(map(list, zip(*dataPairs)))