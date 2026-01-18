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
class CharStringsCompiler(GlobalSubrsCompiler):
    """Helper class for writing the `CharStrings INDEX <https://docs.microsoft.com/en-us/typography/opentype/spec/cff2#9-local-and-global-subr-indexes>`_
    to binary."""

    def getItems(self, items, strings):
        out = []
        for cs in items:
            cs.compile(self.isCFF2)
            out.append(cs.bytecode)
        return out

    def setPos(self, pos, endPos):
        self.parent.rawDict['CharStrings'] = pos