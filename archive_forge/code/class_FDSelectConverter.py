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
class FDSelectConverter(SimpleConverter):

    def _read(self, parent, value):
        file = parent.file
        file.seek(value)
        fdSelect = FDSelect(file, parent.numGlyphs)
        return fdSelect

    def write(self, parent, value):
        return 0

    def xmlWrite(self, xmlWriter, name, value):
        xmlWriter.simpletag(name, [('format', value.format)])
        xmlWriter.newline()

    def xmlRead(self, name, attrs, content, parent):
        fmt = safeEval(attrs['format'])
        file = None
        numGlyphs = None
        fdSelect = FDSelect(file, numGlyphs, fmt)
        return fdSelect