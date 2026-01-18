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
class ULong(IntValue):
    staticSize = 4

    def read(self, reader, font, tableDict):
        return reader.readULong()

    def readArray(self, reader, font, tableDict, count):
        return reader.readULongArray(count)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer.writeULong(value)

    def writeArray(self, writer, font, tableDict, values):
        writer.writeULongArray(values)