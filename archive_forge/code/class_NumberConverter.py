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
class NumberConverter(SimpleConverter):

    def xmlWrite(self, xmlWriter, name, value):
        if isinstance(value, list):
            xmlWriter.begintag(name)
            xmlWriter.newline()
            xmlWriter.indent()
            blendValue = ' '.join([str(val) for val in value])
            xmlWriter.simpletag(kBlendDictOpName, value=blendValue)
            xmlWriter.newline()
            xmlWriter.dedent()
            xmlWriter.endtag(name)
            xmlWriter.newline()
        else:
            xmlWriter.simpletag(name, value=value)
            xmlWriter.newline()

    def xmlRead(self, name, attrs, content, parent):
        valueString = attrs.get('value', None)
        if valueString is None:
            value = parseBlendList(content)
        else:
            value = parseNum(attrs['value'])
        return value