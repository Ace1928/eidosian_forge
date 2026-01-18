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
class BaseFixedValue(FloatValue):
    staticSize = NotImplemented
    precisionBits = NotImplemented
    readerMethod = NotImplemented
    writerMethod = NotImplemented

    def read(self, reader, font, tableDict):
        return self.fromInt(getattr(reader, self.readerMethod)())

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        getattr(writer, self.writerMethod)(self.toInt(value))

    @classmethod
    def fromInt(cls, value):
        return fi2fl(value, cls.precisionBits)

    @classmethod
    def toInt(cls, value):
        return fl2fi(value, cls.precisionBits)

    @classmethod
    def fromString(cls, value):
        return str2fl(value, cls.precisionBits)

    @classmethod
    def toString(cls, value):
        return fl2str(value, cls.precisionBits)