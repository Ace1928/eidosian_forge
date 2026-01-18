import os
import copy
import enum
from operator import ior
import logging
from fontTools.colorLib.builder import MAX_PAINT_COLR_LAYER_COUNT, LayerReuseCache
from fontTools.misc import classifyTools
from fontTools.misc.roundTools import otRound
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables import otBase as otBase
from fontTools.ttLib.tables.otConverters import BaseFixedValue
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.ttLib.tables.DefaultTable import DefaultTable
from fontTools.varLib import builder, models, varStore
from fontTools.varLib.models import nonNone, allNone, allEqual, allEqualTo, subList
from fontTools.varLib.varStore import VarStoreInstancer
from functools import reduce
from fontTools.otlLib.builder import buildSinglePos
from fontTools.otlLib.optimize.gpos import (
from .errors import (
def _Lookup_PairPosFormat2_subtables_flatten(lst, font):
    assert allEqual([l.ValueFormat2 == 0 for l in lst if l.Class1Record]), 'Report bug against fonttools.'
    self = ot.PairPos()
    self.Format = 2
    self.Coverage = ot.Coverage()
    self.ValueFormat1 = reduce(int.__or__, [l.ValueFormat1 for l in lst], 0)
    self.ValueFormat2 = reduce(int.__or__, [l.ValueFormat2 for l in lst], 0)
    glyphs, _ = _merge_GlyphOrders(font, [v.Coverage.glyphs for v in lst])
    self.Coverage.glyphs = glyphs
    matrices = _PairPosFormat2_align_matrices(self, lst, font, transparent=True)
    matrix = self.Class1Record = []
    for rows in zip(*matrices):
        row = ot.Class1Record()
        matrix.append(row)
        row.Class2Record = []
        row = row.Class2Record
        for cols in zip(*list((r.Class2Record for r in rows))):
            col = next(iter((c for c in cols if c is not None)))
            row.append(col)
    return self