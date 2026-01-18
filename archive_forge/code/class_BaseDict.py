from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
class BaseDict(object):

    def __init__(self, strings=None, file=None, offset=None, isCFF2=None):
        assert (isCFF2 is None) == (file is None)
        self.rawDict = {}
        self.skipNames = []
        self.strings = strings
        if file is None:
            return
        self._isCFF2 = isCFF2
        self.file = file
        if offset is not None:
            log.log(DEBUG, 'loading %s at %s', self.__class__.__name__, offset)
            self.offset = offset

    def decompile(self, data):
        log.log(DEBUG, '    length %s is %d', self.__class__.__name__, len(data))
        dec = self.decompilerClass(self.strings, self)
        dec.decompile(data)
        self.rawDict = dec.getDict()
        self.postDecompile()

    def postDecompile(self):
        pass

    def getCompiler(self, strings, parent, isCFF2=None):
        return self.compilerClass(self, strings, parent, isCFF2=isCFF2)

    def __getattr__(self, name):
        if name[:2] == name[-2:] == '__':
            raise AttributeError(name)
        value = self.rawDict.get(name, None)
        if value is None:
            value = self.defaults.get(name)
        if value is None:
            raise AttributeError(name)
        conv = self.converters[name]
        value = conv.read(self, value)
        setattr(self, name, value)
        return value

    def toXML(self, xmlWriter):
        for name in self.order:
            if name in self.skipNames:
                continue
            value = getattr(self, name, None)
            if value is None and name != 'charset':
                continue
            conv = self.converters[name]
            conv.xmlWrite(xmlWriter, name, value)
        ignoredNames = set(self.rawDict) - set(self.order)
        if ignoredNames:
            xmlWriter.comment('some keys were ignored: %s' % ' '.join(sorted(ignoredNames)))
            xmlWriter.newline()

    def fromXML(self, name, attrs, content):
        conv = self.converters[name]
        value = conv.xmlRead(name, attrs, content, self)
        setattr(self, name, value)