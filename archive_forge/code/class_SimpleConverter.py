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
class SimpleConverter(object):

    def read(self, parent, value):
        if not hasattr(parent, 'file'):
            return self._read(parent, value)
        file = parent.file
        pos = file.tell()
        try:
            return self._read(parent, value)
        finally:
            file.seek(pos)

    def _read(self, parent, value):
        return value

    def write(self, parent, value):
        return value

    def xmlWrite(self, xmlWriter, name, value):
        xmlWriter.simpletag(name, value=value)
        xmlWriter.newline()

    def xmlRead(self, name, attrs, content, parent):
        return attrs['value']