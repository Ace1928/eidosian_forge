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
def _setTextDirectionFromCoverageFlags(self, flags, subtable):
    if flags & 32 != 0:
        subtable.TextDirection = 'Any'
    elif flags & 128 != 0:
        subtable.TextDirection = 'Vertical'
    else:
        subtable.TextDirection = 'Horizontal'