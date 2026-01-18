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
class CharsetCompiler(object):

    def __init__(self, strings, charset, parent):
        assert charset[0] == '.notdef'
        isCID = hasattr(parent.dictObj, 'ROS')
        data0 = packCharset0(charset, isCID, strings)
        data = packCharset(charset, isCID, strings)
        if len(data) < len(data0):
            self.data = data
        else:
            self.data = data0
        self.parent = parent

    def setPos(self, pos, endPos):
        self.parent.rawDict['charset'] = pos

    def getDataLength(self):
        return len(self.data)

    def toFile(self, file):
        file.write(self.data)