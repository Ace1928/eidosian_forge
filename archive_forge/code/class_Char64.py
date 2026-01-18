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
class Char64(SimpleValue):
    """An ASCII string with up to 64 characters.

    Unused character positions are filled with 0x00 bytes.
    Used in Apple AAT fonts in the `gcid` table.
    """
    staticSize = 64

    def read(self, reader, font, tableDict):
        data = reader.readData(self.staticSize)
        zeroPos = data.find(b'\x00')
        if zeroPos >= 0:
            data = data[:zeroPos]
        s = tostr(data, encoding='ascii', errors='replace')
        if s != tostr(data, encoding='ascii', errors='ignore'):
            log.warning('replaced non-ASCII characters in "%s"' % s)
        return s

    def write(self, writer, font, tableDict, value, repeatIndex=None):
        data = tobytes(value, encoding='ascii', errors='replace')
        if data != tobytes(value, encoding='ascii', errors='ignore'):
            log.warning('replacing non-ASCII characters in "%s"' % value)
        if len(data) > self.staticSize:
            log.warning('truncating overlong "%s" to %d bytes' % (value, self.staticSize))
        data = (data + b'\x00' * self.staticSize)[:self.staticSize]
        writer.writeData(data)