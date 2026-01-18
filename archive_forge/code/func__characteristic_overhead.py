from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
@staticmethod
def _characteristic_overhead(columns):
    """Returns overhead in bytes of encoding this characteristic
        as a VarData."""
    c = 4 + 6
    c += bit_count(columns) * 2
    return c