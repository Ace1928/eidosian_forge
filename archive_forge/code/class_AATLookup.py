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
class AATLookup(BaseConverter):
    BIN_SEARCH_HEADER_SIZE = 10

    def __init__(self, name, repeat, aux, tableClass, *, description=''):
        BaseConverter.__init__(self, name, repeat, aux, tableClass, description=description)
        if issubclass(self.tableClass, SimpleValue):
            self.converter = self.tableClass(name='Value', repeat=None, aux=None)
        else:
            self.converter = Table(name='Value', repeat=None, aux=None, tableClass=self.tableClass)

    def read(self, reader, font, tableDict):
        format = reader.readUShort()
        if format == 0:
            return self.readFormat0(reader, font)
        elif format == 2:
            return self.readFormat2(reader, font)
        elif format == 4:
            return self.readFormat4(reader, font)
        elif format == 6:
            return self.readFormat6(reader, font)
        elif format == 8:
            return self.readFormat8(reader, font)
        else:
            assert False, 'unsupported lookup format: %d' % format

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        values = list(sorted([(font.getGlyphID(glyph), val) for glyph, val in value.items()]))
        formats = list(sorted(filter(None, [self.buildFormat0(writer, font, values), self.buildFormat2(writer, font, values), self.buildFormat6(writer, font, values), self.buildFormat8(writer, font, values)])))
        dataSize, lookupFormat, writeMethod = formats[0]
        pos = writer.getDataLength()
        writeMethod()
        actualSize = writer.getDataLength() - pos
        assert actualSize == dataSize, 'AATLookup format %d claimed to write %d bytes, but wrote %d' % (lookupFormat, dataSize, actualSize)

    @staticmethod
    def writeBinSearchHeader(writer, numUnits, unitSize):
        writer.writeUShort(unitSize)
        writer.writeUShort(numUnits)
        searchRange, entrySelector, rangeShift = getSearchRange(n=numUnits, itemSize=unitSize)
        writer.writeUShort(searchRange)
        writer.writeUShort(entrySelector)
        writer.writeUShort(rangeShift)

    def buildFormat0(self, writer, font, values):
        numGlyphs = len(font.getGlyphOrder())
        if len(values) != numGlyphs:
            return None
        valueSize = self.converter.staticSize
        return (2 + numGlyphs * valueSize, 0, lambda: self.writeFormat0(writer, font, values))

    def writeFormat0(self, writer, font, values):
        writer.writeUShort(0)
        for glyphID_, value in values:
            self.converter.write(writer, font, tableDict=None, value=value, repeatIndex=None)

    def buildFormat2(self, writer, font, values):
        segStart, segValue = values[0]
        segEnd = segStart
        segments = []
        for glyphID, curValue in values[1:]:
            if glyphID != segEnd + 1 or curValue != segValue:
                segments.append((segStart, segEnd, segValue))
                segStart = segEnd = glyphID
                segValue = curValue
            else:
                segEnd = glyphID
        segments.append((segStart, segEnd, segValue))
        valueSize = self.converter.staticSize
        numUnits, unitSize = (len(segments) + 1, valueSize + 4)
        return (2 + self.BIN_SEARCH_HEADER_SIZE + numUnits * unitSize, 2, lambda: self.writeFormat2(writer, font, segments))

    def writeFormat2(self, writer, font, segments):
        writer.writeUShort(2)
        valueSize = self.converter.staticSize
        numUnits, unitSize = (len(segments), valueSize + 4)
        self.writeBinSearchHeader(writer, numUnits, unitSize)
        for firstGlyph, lastGlyph, value in segments:
            writer.writeUShort(lastGlyph)
            writer.writeUShort(firstGlyph)
            self.converter.write(writer, font, tableDict=None, value=value, repeatIndex=None)
        writer.writeUShort(65535)
        writer.writeUShort(65535)
        writer.writeData(b'\x00' * valueSize)

    def buildFormat6(self, writer, font, values):
        valueSize = self.converter.staticSize
        numUnits, unitSize = (len(values), valueSize + 2)
        return (2 + self.BIN_SEARCH_HEADER_SIZE + (numUnits + 1) * unitSize, 6, lambda: self.writeFormat6(writer, font, values))

    def writeFormat6(self, writer, font, values):
        writer.writeUShort(6)
        valueSize = self.converter.staticSize
        numUnits, unitSize = (len(values), valueSize + 2)
        self.writeBinSearchHeader(writer, numUnits, unitSize)
        for glyphID, value in values:
            writer.writeUShort(glyphID)
            self.converter.write(writer, font, tableDict=None, value=value, repeatIndex=None)
        writer.writeUShort(65535)
        writer.writeData(b'\x00' * valueSize)

    def buildFormat8(self, writer, font, values):
        minGlyphID, maxGlyphID = (values[0][0], values[-1][0])
        if len(values) != maxGlyphID - minGlyphID + 1:
            return None
        valueSize = self.converter.staticSize
        return (6 + len(values) * valueSize, 8, lambda: self.writeFormat8(writer, font, values))

    def writeFormat8(self, writer, font, values):
        firstGlyphID = values[0][0]
        writer.writeUShort(8)
        writer.writeUShort(firstGlyphID)
        writer.writeUShort(len(values))
        for _, value in values:
            self.converter.write(writer, font, tableDict=None, value=value, repeatIndex=None)

    def readFormat0(self, reader, font):
        numGlyphs = len(font.getGlyphOrder())
        data = self.converter.readArray(reader, font, tableDict=None, count=numGlyphs)
        return {font.getGlyphName(k): value for k, value in enumerate(data)}

    def readFormat2(self, reader, font):
        mapping = {}
        pos = reader.pos - 2
        unitSize, numUnits = (reader.readUShort(), reader.readUShort())
        assert unitSize >= 4 + self.converter.staticSize, unitSize
        for i in range(numUnits):
            reader.seek(pos + i * unitSize + 12)
            last = reader.readUShort()
            first = reader.readUShort()
            value = self.converter.read(reader, font, tableDict=None)
            if last != 65535:
                for k in range(first, last + 1):
                    mapping[font.getGlyphName(k)] = value
        return mapping

    def readFormat4(self, reader, font):
        mapping = {}
        pos = reader.pos - 2
        unitSize = reader.readUShort()
        assert unitSize >= 6, unitSize
        for i in range(reader.readUShort()):
            reader.seek(pos + i * unitSize + 12)
            last = reader.readUShort()
            first = reader.readUShort()
            offset = reader.readUShort()
            if last != 65535:
                dataReader = reader.getSubReader(0)
                dataReader.seek(pos + offset)
                data = self.converter.readArray(dataReader, font, tableDict=None, count=last - first + 1)
                for k, v in enumerate(data):
                    mapping[font.getGlyphName(first + k)] = v
        return mapping

    def readFormat6(self, reader, font):
        mapping = {}
        pos = reader.pos - 2
        unitSize = reader.readUShort()
        assert unitSize >= 2 + self.converter.staticSize, unitSize
        for i in range(reader.readUShort()):
            reader.seek(pos + i * unitSize + 12)
            glyphID = reader.readUShort()
            value = self.converter.read(reader, font, tableDict=None)
            if glyphID != 65535:
                mapping[font.getGlyphName(glyphID)] = value
        return mapping

    def readFormat8(self, reader, font):
        first = reader.readUShort()
        count = reader.readUShort()
        data = self.converter.readArray(reader, font, tableDict=None, count=count)
        return {font.getGlyphName(first + k): value for k, value in enumerate(data)}

    def xmlRead(self, attrs, content, font):
        value = {}
        for element in content:
            if isinstance(element, tuple):
                name, a, eltContent = element
                if name == 'Lookup':
                    value[a['glyph']] = self.converter.xmlRead(a, eltContent, font)
        return value

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.begintag(name, attrs)
        xmlWriter.newline()
        for glyph, value in sorted(value.items()):
            self.converter.xmlWrite(xmlWriter, font, value=value, name='Lookup', attrs=[('glyph', glyph)])
        xmlWriter.endtag(name)
        xmlWriter.newline()