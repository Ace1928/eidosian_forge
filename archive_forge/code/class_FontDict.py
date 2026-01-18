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
class FontDict(BaseDict):
    defaults = {}
    converters = buildConverters(topDictOperators)
    compilerClass = FontDictCompiler
    orderCFF = ['FontName', 'FontMatrix', 'Weight', 'Private']
    orderCFF2 = ['Private']
    decompilerClass = TopDictDecompiler

    def __init__(self, strings=None, file=None, offset=None, GlobalSubrs=None, isCFF2=None, vstore=None):
        super(FontDict, self).__init__(strings, file, offset, isCFF2=isCFF2)
        self.vstore = vstore
        self.setCFF2(isCFF2)

    def setCFF2(self, isCFF2):
        if isCFF2:
            self.order = self.orderCFF2
            self._isCFF2 = True
        else:
            self.order = self.orderCFF
            self._isCFF2 = False