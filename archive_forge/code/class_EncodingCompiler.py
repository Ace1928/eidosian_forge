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
class EncodingCompiler(object):

    def __init__(self, strings, encoding, parent):
        assert not isinstance(encoding, str)
        data0 = packEncoding0(parent.dictObj.charset, encoding, parent.strings)
        data1 = packEncoding1(parent.dictObj.charset, encoding, parent.strings)
        if len(data0) < len(data1):
            self.data = data0
        else:
            self.data = data1
        self.parent = parent

    def setPos(self, pos, endPos):
        self.parent.rawDict['Encoding'] = pos

    def getDataLength(self):
        return len(self.data)

    def toFile(self, file):
        file.write(self.data)