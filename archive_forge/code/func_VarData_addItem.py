from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def VarData_addItem(self, deltas, *, round=round):
    deltas = [round(d) for d in deltas]
    countUs = self.VarRegionCount
    countThem = len(deltas)
    if countUs + 1 == countThem:
        deltas = list(deltas[1:])
    else:
        assert countUs == countThem, (countUs, countThem)
        deltas = list(deltas)
    self.Item.append(deltas)
    self.ItemCount = len(self.Item)