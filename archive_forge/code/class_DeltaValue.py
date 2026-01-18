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
class DeltaValue(BaseConverter):

    def read(self, reader, font, tableDict):
        StartSize = tableDict['StartSize']
        EndSize = tableDict['EndSize']
        DeltaFormat = tableDict['DeltaFormat']
        assert DeltaFormat in (1, 2, 3), 'illegal DeltaFormat'
        nItems = EndSize - StartSize + 1
        nBits = 1 << DeltaFormat
        minusOffset = 1 << nBits
        mask = (1 << nBits) - 1
        signMask = 1 << nBits - 1
        DeltaValue = []
        tmp, shift = (0, 0)
        for i in range(nItems):
            if shift == 0:
                tmp, shift = (reader.readUShort(), 16)
            shift = shift - nBits
            value = tmp >> shift & mask
            if value & signMask:
                value = value - minusOffset
            DeltaValue.append(value)
        return DeltaValue

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        StartSize = tableDict['StartSize']
        EndSize = tableDict['EndSize']
        DeltaFormat = tableDict['DeltaFormat']
        DeltaValue = value
        assert DeltaFormat in (1, 2, 3), 'illegal DeltaFormat'
        nItems = EndSize - StartSize + 1
        nBits = 1 << DeltaFormat
        assert len(DeltaValue) == nItems
        mask = (1 << nBits) - 1
        tmp, shift = (0, 16)
        for value in DeltaValue:
            shift = shift - nBits
            tmp = tmp | (value & mask) << shift
            if shift == 0:
                writer.writeUShort(tmp)
                tmp, shift = (0, 16)
        if shift != 16:
            writer.writeUShort(tmp)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [('value', value)])
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        return safeEval(attrs['value'])