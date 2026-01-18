from fontTools import config
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables
from fontTools.ttLib.tables.otBase import USE_HARFBUZZ_REPACKER
from fontTools.otlLib.maxContextCalc import maxCtxFont
from fontTools.pens.basePen import NullPen
from fontTools.misc.loggingTools import Timer
from fontTools.misc.cliTools import makeOutputFileName
from fontTools.subset.util import _add_method, _uniq_sort
from fontTools.subset.cff import *
from fontTools.subset.svg import *
from fontTools.varLib import varStore  # for subset_varidxes
from fontTools.ttLib.tables._n_a_m_e import NameRecordVisitor
import sys
import struct
import array
import logging
from collections import Counter, defaultdict
from functools import reduce
from types import MethodType
def _remap_index_map(s, varidx_map, table_map):
    map_ = {k: varidx_map[v] for k, v in table_map.mapping.items()}
    last_idx = varidx_map[table_map.mapping[s.last_retained_glyph]]
    for g, i in s.reverseEmptiedGlyphMap.items():
        map_[g] = last_idx if i > s.last_retained_order else 0
    return map_