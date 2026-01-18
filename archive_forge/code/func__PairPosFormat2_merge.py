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
def _PairPosFormat2_merge(self, lst, merger):
    assert allEqual([l.ValueFormat2 == 0 for l in lst if l.Class1Record]), 'Report bug against fonttools.'
    merger.mergeObjects(self, lst, exclude=('Coverage', 'ClassDef1', 'Class1Count', 'ClassDef2', 'Class2Count', 'Class1Record', 'ValueFormat1', 'ValueFormat2'))
    glyphs, _ = _merge_GlyphOrders(merger.font, [v.Coverage.glyphs for v in lst])
    self.Coverage.glyphs = glyphs
    for l, subtables in zip(lst, merger.lookup_subtables):
        if l.Coverage.glyphs != glyphs:
            assert l == subtables[-1]
    matrices = _PairPosFormat2_align_matrices(self, lst, merger.font)
    self.Class1Record = list(matrices[0])
    merger.mergeLists(self.Class1Record, matrices)