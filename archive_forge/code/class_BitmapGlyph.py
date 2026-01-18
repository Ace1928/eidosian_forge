from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
class BitmapGlyph(object):
    fileExtension = '.bin'
    xmlDataFunctions = {'raw': (_writeRawImageData, _readRawImageData), 'row': (_writeRowImageData, _readRowImageData), 'bitwise': (_writeBitwiseImageData, _readBitwiseImageData), 'extfile': (_writeExtFileImageData, _readExtFileImageData)}

    def __init__(self, data, ttFont):
        self.data = data
        self.ttFont = ttFont

    def __getattr__(self, attr):
        if attr[:2] == '__':
            raise AttributeError(attr)
        if attr == 'data':
            raise AttributeError(attr)
        self.decompile()
        del self.data
        return getattr(self, attr)

    def ensureDecompiled(self, recurse=False):
        if hasattr(self, 'data'):
            self.decompile()
            del self.data

    def getFormat(self):
        return safeEval(self.__class__.__name__[len(_bitmapGlyphSubclassPrefix):])

    def toXML(self, strikeIndex, glyphName, writer, ttFont):
        writer.begintag(self.__class__.__name__, [('name', glyphName)])
        writer.newline()
        self.writeMetrics(writer, ttFont)
        self.writeData(strikeIndex, glyphName, writer, ttFont)
        writer.endtag(self.__class__.__name__)
        writer.newline()

    def fromXML(self, name, attrs, content, ttFont):
        self.readMetrics(name, attrs, content, ttFont)
        for element in content:
            if not isinstance(element, tuple):
                continue
            name, attr, content = element
            if not name.endswith('imagedata'):
                continue
            option = name[:-len('imagedata')]
            assert option in self.__class__.xmlDataFunctions
            self.readData(name, attr, content, ttFont)

    def writeMetrics(self, writer, ttFont):
        pass

    def readMetrics(self, name, attrs, content, ttFont):
        pass

    def writeData(self, strikeIndex, glyphName, writer, ttFont):
        try:
            writeFunc, readFunc = self.__class__.xmlDataFunctions[ttFont.bitmapGlyphDataFormat]
        except KeyError:
            writeFunc = _writeRawImageData
        writeFunc(strikeIndex, glyphName, self, writer, ttFont)

    def readData(self, name, attrs, content, ttFont):
        option = name[:-len('imagedata')]
        writeFunc, readFunc = self.__class__.xmlDataFunctions[option]
        readFunc(self, name, attrs, content, ttFont)