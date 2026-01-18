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
class Version(SimpleValue):
    staticSize = 4

    def read(self, reader, font, tableDict):
        value = reader.readLong()
        return value

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        value = fi2ve(value)
        writer.writeLong(value)

    @staticmethod
    def fromString(value):
        return ve2fi(value)

    @staticmethod
    def toString(value):
        return '0x%08x' % value

    @staticmethod
    def fromFloat(v):
        return fl2fi(v, 16)