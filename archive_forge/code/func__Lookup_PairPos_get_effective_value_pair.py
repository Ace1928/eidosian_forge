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
def _Lookup_PairPos_get_effective_value_pair(merger, subtables, firstGlyph, secondGlyph):
    for self in subtables:
        if self is None or type(self) != ot.PairPos or self.Coverage is None or (firstGlyph not in self.Coverage.glyphs):
            continue
        if self.Format == 1:
            ps = self.PairSet[self.Coverage.glyphs.index(firstGlyph)]
            pvr = ps.PairValueRecord
            for rec in pvr:
                if rec.SecondGlyph == secondGlyph:
                    return rec
            continue
        elif self.Format == 2:
            klass1 = self.ClassDef1.classDefs.get(firstGlyph, 0)
            klass2 = self.ClassDef2.classDefs.get(secondGlyph, 0)
            return self.Class1Record[klass1].Class2Record[klass2]
        else:
            raise UnsupportedFormat(merger, subtable='pair positioning lookup')
    return None