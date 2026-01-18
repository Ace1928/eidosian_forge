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
class VarIdxMapValue(BaseConverter):

    def read(self, reader, font, tableDict):
        fmt = tableDict['EntryFormat']
        nItems = tableDict['MappingCount']
        innerBits = 1 + (fmt & 15)
        innerMask = (1 << innerBits) - 1
        outerMask = 4294967295 - innerMask
        outerShift = 16 - innerBits
        entrySize = 1 + ((fmt & 48) >> 4)
        readArray = {1: reader.readUInt8Array, 2: reader.readUShortArray, 3: reader.readUInt24Array, 4: reader.readULongArray}[entrySize]
        return [(raw & outerMask) << outerShift | raw & innerMask for raw in readArray(nItems)]

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        fmt = tableDict['EntryFormat']
        mapping = value
        writer['MappingCount'].setValue(len(mapping))
        innerBits = 1 + (fmt & 15)
        innerMask = (1 << innerBits) - 1
        outerShift = 16 - innerBits
        entrySize = 1 + ((fmt & 48) >> 4)
        writeArray = {1: writer.writeUInt8Array, 2: writer.writeUShortArray, 3: writer.writeUInt24Array, 4: writer.writeULongArray}[entrySize]
        writeArray([(idx & 4294901760) >> outerShift | idx & innerMask for idx in mapping])