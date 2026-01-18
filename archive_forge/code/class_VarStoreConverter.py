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
class VarStoreConverter(SimpleConverter):

    def _read(self, parent, value):
        file = parent.file
        file.seek(value)
        varStore = VarStoreData(file)
        varStore.decompile()
        return varStore

    def write(self, parent, value):
        return 0

    def xmlWrite(self, xmlWriter, name, value):
        value.writeXML(xmlWriter, name)

    def xmlRead(self, name, attrs, content, parent):
        varStore = VarStoreData()
        varStore.xmlRead(name, attrs, content, parent)
        return varStore