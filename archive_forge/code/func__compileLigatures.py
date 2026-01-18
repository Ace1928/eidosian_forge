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
def _compileLigatures(self, table, font):
    if not hasattr(table, 'Ligatures'):
        return None
    writer = OTTableWriter()
    for glyphName in table.Ligatures:
        writer.writeUShort(font.getGlyphID(glyphName))
    return writer.getAllData()