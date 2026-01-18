from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
class AATLookupWithDataOffset(BaseConverter):

    def read(self, reader, font, tableDict):
        lookupOffset = reader.readULong()
        dataOffset = reader.readULong()
        lookupReader = reader.getSubReader(lookupOffset)
        lookup = AATLookup('DataOffsets', None, None, UShort)
        offsets = lookup.read(lookupReader, font, tableDict)
        result = {}
        for glyph, offset in offsets.items():
            dataReader = reader.getSubReader(offset + dataOffset)
            item = self.tableClass()
            item.decompile(dataReader, font)
            result[glyph] = item
        return result

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        offsetByGlyph, offsetByData, dataLen = ({}, {}, 0)
        compiledData = []
        for glyph in sorted(value, key=font.getGlyphID):
            subWriter = OTTableWriter()
            value[glyph].compile(subWriter, font)
            data = subWriter.getAllData()
            offset = offsetByData.get(data, None)
            if offset == None:
                offset = dataLen
                dataLen = dataLen + len(data)
                offsetByData[data] = offset
                compiledData.append(data)
            offsetByGlyph[glyph] = offset
        lookupWriter = writer.getSubWriter()
        lookup = AATLookup('DataOffsets', None, None, UShort)
        lookup.write(lookupWriter, font, tableDict, offsetByGlyph, None)
        dataWriter = writer.getSubWriter()
        writer.writeSubTable(lookupWriter, offsetSize=4)
        writer.writeSubTable(dataWriter, offsetSize=4)
        for d in compiledData:
            dataWriter.writeData(d)

    def xmlRead(self, attrs, content, font):
        lookup = AATLookup('DataOffsets', None, None, self.tableClass)
        return lookup.xmlRead(attrs, content, font)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        lookup = AATLookup('DataOffsets', None, None, self.tableClass)
        lookup.xmlWrite(xmlWriter, font, value, name, attrs)