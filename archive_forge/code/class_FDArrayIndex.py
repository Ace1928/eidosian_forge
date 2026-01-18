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
class FDArrayIndex(Index):
    compilerClass = FDArrayIndexCompiler

    def toXML(self, xmlWriter):
        for i in range(len(self)):
            xmlWriter.begintag('FontDict', index=i)
            xmlWriter.newline()
            self[i].toXML(xmlWriter)
            xmlWriter.endtag('FontDict')
            xmlWriter.newline()

    def produceItem(self, index, data, file, offset):
        fontDict = FontDict(self.strings, file, offset, self.GlobalSubrs, isCFF2=self._isCFF2, vstore=self.vstore)
        fontDict.decompile(data)
        return fontDict

    def fromXML(self, name, attrs, content):
        if name != 'FontDict':
            return
        fontDict = FontDict()
        for element in content:
            if isinstance(element, str):
                continue
            name, attrs, content = element
            fontDict.fromXML(name, attrs, content)
        self.append(fontDict)