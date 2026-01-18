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
class StructWithLength(Struct):

    def read(self, reader, font, tableDict):
        pos = reader.pos
        table = self.tableClass()
        table.decompile(reader, font)
        reader.seek(pos + table.StructLength)
        return table

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        for convIndex, conv in enumerate(value.getConverters()):
            if conv.name == 'StructLength':
                break
        lengthIndex = len(writer.items) + convIndex
        if isinstance(value, FormatSwitchingBaseTable):
            lengthIndex += 1
        deadbeef = {1: 222, 2: 57005, 4: 3735928559}[conv.staticSize]
        before = writer.getDataLength()
        value.StructLength = deadbeef
        value.compile(writer, font)
        length = writer.getDataLength() - before
        lengthWriter = writer.getSubWriter()
        conv.write(lengthWriter, font, tableDict, length)
        assert writer.items[lengthIndex] == b'\xde\xad\xbe\xef'[:conv.staticSize]
        writer.items[lengthIndex] = lengthWriter.getAllData()