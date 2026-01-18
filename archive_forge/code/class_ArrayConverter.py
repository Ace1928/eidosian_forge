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
class ArrayConverter(SimpleConverter):

    def xmlWrite(self, xmlWriter, name, value):
        if value and isinstance(value[0], list):
            xmlWriter.begintag(name)
            xmlWriter.newline()
            xmlWriter.indent()
            for valueList in value:
                blendValue = ' '.join([str(val) for val in valueList])
                xmlWriter.simpletag(kBlendDictOpName, value=blendValue)
                xmlWriter.newline()
            xmlWriter.dedent()
            xmlWriter.endtag(name)
            xmlWriter.newline()
        else:
            value = ' '.join([str(val) for val in value])
            xmlWriter.simpletag(name, value=value)
            xmlWriter.newline()

    def xmlRead(self, name, attrs, content, parent):
        valueString = attrs.get('value', None)
        if valueString is None:
            valueList = parseBlendList(content)
        else:
            values = valueString.split()
            valueList = [parseNum(value) for value in values]
        return valueList