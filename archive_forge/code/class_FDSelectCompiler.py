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
class FDSelectCompiler(object):

    def __init__(self, fdSelect, parent):
        fmt = fdSelect.format
        fdSelectArray = fdSelect.gidArray
        if fmt == 0:
            self.data = packFDSelect0(fdSelectArray)
        elif fmt == 3:
            self.data = packFDSelect3(fdSelectArray)
        elif fmt == 4:
            self.data = packFDSelect4(fdSelectArray)
        else:
            data0 = packFDSelect0(fdSelectArray)
            data3 = packFDSelect3(fdSelectArray)
            if len(data0) < len(data3):
                self.data = data0
                fdSelect.format = 0
            else:
                self.data = data3
                fdSelect.format = 3
        self.parent = parent

    def setPos(self, pos, endPos):
        self.parent.rawDict['FDSelect'] = pos

    def getDataLength(self):
        return len(self.data)

    def toFile(self, file):
        file.write(self.data)