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
class PrivateDictConverter(TableConverter):

    def getClass(self):
        return PrivateDict

    def _read(self, parent, value):
        size, offset = value
        file = parent.file
        isCFF2 = parent._isCFF2
        try:
            vstore = parent.vstore
        except AttributeError:
            vstore = None
        priv = PrivateDict(parent.strings, file, offset, isCFF2=isCFF2, vstore=vstore)
        file.seek(offset)
        data = file.read(size)
        assert len(data) == size
        priv.decompile(data)
        return priv

    def write(self, parent, value):
        return (0, 0)