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
class Table(Struct):
    staticSize = 2

    def readOffset(self, reader):
        return reader.readUShort()

    def writeNullOffset(self, writer):
        writer.writeUShort(0)

    def read(self, reader, font, tableDict):
        offset = self.readOffset(reader)
        if offset == 0:
            return None
        table = self.tableClass()
        reader = reader.getSubReader(offset)
        if font.lazy:
            table.reader = reader
            table.font = font
        else:
            table.decompile(reader, font)
        return table

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        if value is None:
            self.writeNullOffset(writer)
        else:
            subWriter = writer.getSubWriter()
            subWriter.name = self.name
            if repeatIndex is not None:
                subWriter.repeatIndex = repeatIndex
            writer.writeSubTable(subWriter, offsetSize=self.staticSize)
            value.compile(subWriter, font)