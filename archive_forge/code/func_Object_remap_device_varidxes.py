from fontTools.misc.roundTools import noRound, otRound
from fontTools.misc.intTools import bit_count
from fontTools.ttLib.tables import otTables as ot
from fontTools.varLib.models import supportScalar
from fontTools.varLib.builder import (
from functools import partial
from collections import defaultdict
from heapq import heappush, heappop
def Object_remap_device_varidxes(self, varidxes_map):
    mapper = partial(_Device_mapVarIdx, mapping=varidxes_map, done=set())
    _visit(self, mapper)