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
class _UInt8Enum(UInt8):
    enumClass = NotImplemented

    def read(self, reader, font, tableDict):
        return self.enumClass(super().read(reader, font, tableDict))

    @classmethod
    def fromString(cls, value):
        return getattr(cls.enumClass, value.upper())

    @classmethod
    def toString(cls, value):
        return cls.enumClass(value).name.lower()