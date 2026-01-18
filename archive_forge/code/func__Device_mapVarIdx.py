from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def _Device_mapVarIdx(self, mapping, done):
    """Map VarIdx in this Device table (if any) through mapping."""
    if id(self) in done:
        return
    done.add(id(self))
    if self.DeltaFormat == 32768:
        varIdx = mapping[(self.StartSize << 16) + self.EndSize]
        self.StartSize = varIdx >> 16
        self.EndSize = varIdx & 65535