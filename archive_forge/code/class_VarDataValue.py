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
class VarDataValue(BaseConverter):

    def read(self, reader, font, tableDict):
        values = []
        regionCount = tableDict['VarRegionCount']
        wordCount = tableDict['NumShorts']
        longWords = bool(wordCount & 32768)
        wordCount = wordCount & 32767
        if longWords:
            readBigArray, readSmallArray = (reader.readLongArray, reader.readShortArray)
        else:
            readBigArray, readSmallArray = (reader.readShortArray, reader.readInt8Array)
        n1, n2 = (min(regionCount, wordCount), max(regionCount, wordCount))
        values.extend(readBigArray(n1))
        values.extend(readSmallArray(n2 - n1))
        if n2 > regionCount:
            del values[regionCount:]
        return values

    def write(self, writer, font, tableDict, values, repeatIndex=None):
        regionCount = tableDict['VarRegionCount']
        wordCount = tableDict['NumShorts']
        longWords = bool(wordCount & 32768)
        wordCount = wordCount & 32767
        writeBigArray, writeSmallArray = {False: (writer.writeShortArray, writer.writeInt8Array), True: (writer.writeLongArray, writer.writeShortArray)}[longWords]
        n1, n2 = (min(regionCount, wordCount), max(regionCount, wordCount))
        writeBigArray(values[:n1])
        writeSmallArray(values[n1:regionCount])
        if n2 > regionCount:
            writer.writeSmallArray([0] * (n2 - regionCount))

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [('value', value)])
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        return safeEval(attrs['value'])