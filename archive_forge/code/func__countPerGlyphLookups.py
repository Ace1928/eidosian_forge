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
def _countPerGlyphLookups(self, table):
    numLookups = 0
    for state in table.States:
        for t in state.Transitions.values():
            if isinstance(t, ContextualMorphAction):
                if t.MarkIndex != 65535:
                    numLookups = max(numLookups, t.MarkIndex + 1)
                if t.CurrentIndex != 65535:
                    numLookups = max(numLookups, t.CurrentIndex + 1)
    return numLookups