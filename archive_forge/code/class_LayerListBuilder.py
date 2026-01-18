import collections
import copy
import enum
from functools import partial
from math import ceil, log
from typing import (
from fontTools.misc.arrayTools import intRect
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_
from fontTools.ttLib.tables import C_P_A_L_
from fontTools.ttLib.tables import _n_a_m_e
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otTables import ExtendMode, CompositeMode
from .errors import ColorLibError
from .geometry import round_start_circle_stable_containment
from .table_builder import BuildCallback, TableBuilder
class LayerListBuilder:
    layers: List[ot.Paint]
    cache: LayerReuseCache
    allowLayerReuse: bool

    def __init__(self, *, allowLayerReuse=True):
        self.layers = []
        if allowLayerReuse:
            self.cache = LayerReuseCache()
        else:
            self.cache = None
        callbacks = _buildPaintCallbacks()
        callbacks[BuildCallback.BEFORE_BUILD, ot.Paint, ot.PaintFormat.PaintColrLayers] = self._beforeBuildPaintColrLayers
        self.tableBuilder = TableBuilder(callbacks)

    def _beforeBuildPaintColrLayers(self, dest, source):
        if isinstance(source.get('NumLayers', None), collections.abc.Sequence):
            layers = source['NumLayers']
        else:
            layers = source['Layers']
        layers = [self.buildPaint(l) for l in layers]
        if len(layers) == 1:
            return (layers[0], {})
        if self.cache is not None:
            layers = self.cache.try_reuse(layers)
        is_tree = len(layers) > MAX_PAINT_COLR_LAYER_COUNT
        layers = build_n_ary_tree(layers, n=MAX_PAINT_COLR_LAYER_COUNT)

        def listToColrLayers(layer):
            if isinstance(layer, collections.abc.Sequence):
                return self.buildPaint({'Format': ot.PaintFormat.PaintColrLayers, 'Layers': [listToColrLayers(l) for l in layer]})
            return layer
        layers = [listToColrLayers(l) for l in layers]
        if len(layers) == 1:
            return (layers[0], {})
        paint = ot.Paint()
        paint.Format = int(ot.PaintFormat.PaintColrLayers)
        paint.NumLayers = len(layers)
        paint.FirstLayerIndex = len(self.layers)
        self.layers.extend(layers)
        if self.cache is not None and (not is_tree):
            self.cache.add(layers, paint.FirstLayerIndex)
        return (paint, {})

    def buildPaint(self, paint: _PaintInput) -> ot.Paint:
        return self.tableBuilder.build(ot.Paint, paint)

    def build(self) -> Optional[ot.LayerList]:
        if not self.layers:
            return None
        layers = ot.LayerList()
        layers.LayerCount = len(self.layers)
        layers.Paint = self.layers
        return layers