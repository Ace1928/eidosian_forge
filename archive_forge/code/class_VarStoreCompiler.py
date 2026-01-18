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
class VarStoreCompiler(object):

    def __init__(self, varStoreData, parent):
        self.parent = parent
        if not varStoreData.data:
            varStoreData.compile()
        data = [packCard16(len(varStoreData.data)), varStoreData.data]
        self.data = bytesjoin(data)

    def setPos(self, pos, endPos):
        self.parent.rawDict['VarStore'] = pos

    def getDataLength(self):
        return len(self.data)

    def toFile(self, file):
        file.write(self.data)