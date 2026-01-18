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
def _merge_PaintColrLayers(self, out, lst):
    out_layers = list(_flatten_layers(out, self.font['COLR'].table))
    assert len(self.ttfs) == len(lst)
    master_layerses = [list(_flatten_layers(lst[i], self.ttfs[i]['COLR'].table)) for i in range(len(lst))]
    try:
        self.mergeLists(out_layers, master_layerses)
    except VarLibMergeError as e:
        e.stack.append('.Layers')
        raise
    if self.layerReuseCache is not None:
        out_layers = self.layerReuseCache.try_reuse(out_layers)
    is_tree = len(out_layers) > MAX_PAINT_COLR_LAYER_COUNT
    out_layers = build_n_ary_tree(out_layers, n=MAX_PAINT_COLR_LAYER_COUNT)

    def listToColrLayers(paint):
        if isinstance(paint, list):
            layers = [listToColrLayers(l) for l in paint]
            paint = ot.Paint()
            paint.Format = int(ot.PaintFormat.PaintColrLayers)
            paint.NumLayers = len(layers)
            paint.FirstLayerIndex = len(self.layers)
            self.layers.extend(layers)
            if self.layerReuseCache is not None:
                self.layerReuseCache.add(layers, paint.FirstLayerIndex)
        return paint
    out_layers = [listToColrLayers(l) for l in out_layers]
    if len(out_layers) == 1 and out_layers[0].Format == ot.PaintFormat.PaintColrLayers:
        out.NumLayers = out_layers[0].NumLayers
        out.FirstLayerIndex = out_layers[0].FirstLayerIndex
    else:
        out.NumLayers = len(out_layers)
        out.FirstLayerIndex = len(self.layers)
        self.layers.extend(out_layers)
        if self.layerReuseCache is not None and (not is_tree):
            self.layerReuseCache.add(out_layers, out.FirstLayerIndex)