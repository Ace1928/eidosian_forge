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
def _PairPosFormat1_merge(self, lst, merger):
    assert allEqual([l.ValueFormat2 == 0 for l in lst if l.PairSet]), 'Report bug against fonttools.'
    merger.mergeObjects(self, lst, exclude=('Coverage', 'PairSet', 'PairSetCount', 'ValueFormat1', 'ValueFormat2'))
    empty = ot.PairSet()
    empty.PairValueRecord = []
    empty.PairValueCount = 0
    glyphs, padded = _merge_GlyphOrders(merger.font, [v.Coverage.glyphs for v in lst], [v.PairSet for v in lst], default=empty)
    self.Coverage.glyphs = glyphs
    self.PairSet = [ot.PairSet() for _ in glyphs]
    self.PairSetCount = len(self.PairSet)
    for glyph, ps in zip(glyphs, self.PairSet):
        ps._firstGlyph = glyph
    merger.mergeLists(self.PairSet, padded)