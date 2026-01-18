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
class SimpleValue(BaseConverter):

    @staticmethod
    def toString(value):
        return value

    @staticmethod
    def fromString(value):
        return value

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        xmlWriter.simpletag(name, attrs + [('value', self.toString(value))])
        xmlWriter.newline()

    def xmlRead(self, attrs, content, font):
        return self.fromString(attrs['value'])