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
def _xmlReadLigatures(self, attrs, content, font):
    ligs = []
    for eltName, eltAttrs, _eltContent in filter(istuple, content):
        if eltName == 'Ligature':
            ligs.append(eltAttrs['glyph'])
    return ligs