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
class TableConverter(SimpleConverter):

    def xmlWrite(self, xmlWriter, name, value):
        xmlWriter.begintag(name)
        xmlWriter.newline()
        value.toXML(xmlWriter)
        xmlWriter.endtag(name)
        xmlWriter.newline()

    def xmlRead(self, name, attrs, content, parent):
        ob = self.getClass()()
        for element in content:
            if isinstance(element, str):
                continue
            name, attrs, content = element
            ob.fromXML(name, attrs, content)
        return ob