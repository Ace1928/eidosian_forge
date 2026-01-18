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
class ValueRecord(ValueFormat):

    def getRecordSize(self, reader):
        return 2 * len(reader[self.which])

    def read(self, reader, font, tableDict):
        return reader[self.which].readValueRecord(reader, font)

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        writer[self.which].writeValueRecord(writer, font, value)

    def xmlWrite(self, xmlWriter, font, value, name, attrs):
        if value is None:
            pass
        else:
            value.toXML(xmlWriter, font, self.name, attrs)

    def xmlRead(self, attrs, content, font):
        from .otBase import ValueRecord
        value = ValueRecord()
        value.fromXML(None, attrs, content, font)
        return value