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
def _MarkBasePosFormat1_merge(self, lst, merger, Mark='Mark', Base='Base'):
    self.ClassCount = max((l.ClassCount for l in lst))
    MarkCoverageGlyphs, MarkRecords = _merge_GlyphOrders(merger.font, [getattr(l, Mark + 'Coverage').glyphs for l in lst], [getattr(l, Mark + 'Array').MarkRecord for l in lst])
    getattr(self, Mark + 'Coverage').glyphs = MarkCoverageGlyphs
    BaseCoverageGlyphs, BaseRecords = _merge_GlyphOrders(merger.font, [getattr(l, Base + 'Coverage').glyphs for l in lst], [getattr(getattr(l, Base + 'Array'), Base + 'Record') for l in lst])
    getattr(self, Base + 'Coverage').glyphs = BaseCoverageGlyphs
    records = []
    for g, glyphRecords in zip(MarkCoverageGlyphs, zip(*MarkRecords)):
        allClasses = [r.Class for r in glyphRecords if r is not None]
        if not allEqual(allClasses):
            raise ShouldBeConstant(merger, expected=allClasses[0], got=allClasses)
        else:
            rec = ot.MarkRecord()
            rec.Class = allClasses[0]
            allAnchors = [None if r is None else r.MarkAnchor for r in glyphRecords]
            if allNone(allAnchors):
                anchor = None
            else:
                anchor = ot.Anchor()
                anchor.Format = 1
                merger.mergeThings(anchor, allAnchors)
            rec.MarkAnchor = anchor
        records.append(rec)
    array = ot.MarkArray()
    array.MarkRecord = records
    array.MarkCount = len(records)
    setattr(self, Mark + 'Array', array)
    records = []
    for g, glyphRecords in zip(BaseCoverageGlyphs, zip(*BaseRecords)):
        if allNone(glyphRecords):
            rec = None
        else:
            rec = getattr(ot, Base + 'Record')()
            anchors = []
            setattr(rec, Base + 'Anchor', anchors)
            glyphAnchors = [[] if r is None else getattr(r, Base + 'Anchor') for r in glyphRecords]
            for l in glyphAnchors:
                l.extend([None] * (self.ClassCount - len(l)))
            for allAnchors in zip(*glyphAnchors):
                if allNone(allAnchors):
                    anchor = None
                else:
                    anchor = ot.Anchor()
                    anchor.Format = 1
                    merger.mergeThings(anchor, allAnchors)
                anchors.append(anchor)
        records.append(rec)
    array = getattr(ot, Base + 'Array')()
    setattr(array, Base + 'Record', records)
    setattr(array, Base + 'Count', len(records))
    setattr(self, Base + 'Array', array)