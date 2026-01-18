from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def _Device_recordVarIdx(self, s):
    """Add VarIdx in this Device table (if any) to the set s."""
    if self.DeltaFormat == 32768:
        s.add((self.StartSize << 16) + self.EndSize)