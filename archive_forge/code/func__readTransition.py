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
def _readTransition(self, reader, entryIndex, font, actionReader):
    transition = self.tableClass()
    entryReader = reader.getSubReader(reader.pos + entryIndex * transition.staticSize)
    transition.decompile(entryReader, font, actionReader)
    return transition