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
class ROSConverter(SimpleConverter):

    def xmlWrite(self, xmlWriter, name, value):
        registry, order, supplement = value
        xmlWriter.simpletag(name, [('Registry', tostr(registry)), ('Order', tostr(order)), ('Supplement', supplement)])
        xmlWriter.newline()

    def xmlRead(self, name, attrs, content, parent):
        return (attrs['Registry'], attrs['Order'], safeEval(attrs['Supplement']))