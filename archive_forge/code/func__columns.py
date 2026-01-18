from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
@staticmethod
def _columns(chars):
    cols = 0
    i = 1
    while chars:
        if chars & 15:
            cols |= i
        chars >>= 4
        i <<= 1
    return cols