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
def _Lookup_SinglePos_subtables_flatten(lst, font, min_inclusive_rec_format):
    glyphs, _ = _merge_GlyphOrders(font, [v.Coverage.glyphs for v in lst], None)
    num_glyphs = len(glyphs)
    new = ot.SinglePos()
    new.Format = 2
    new.ValueFormat = min_inclusive_rec_format
    new.Coverage = ot.Coverage()
    new.Coverage.glyphs = glyphs
    new.ValueCount = num_glyphs
    new.Value = [None] * num_glyphs
    for singlePos in lst:
        if singlePos.Format == 1:
            val_rec = singlePos.Value
            for gname in singlePos.Coverage.glyphs:
                i = glyphs.index(gname)
                new.Value[i] = copy.deepcopy(val_rec)
        elif singlePos.Format == 2:
            for j, gname in enumerate(singlePos.Coverage.glyphs):
                val_rec = singlePos.Value[j]
                i = glyphs.index(gname)
                new.Value[i] = copy.deepcopy(val_rec)
    return [new]