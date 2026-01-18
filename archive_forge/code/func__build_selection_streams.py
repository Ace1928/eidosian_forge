from collections import namedtuple
import numpy as np
import param
from param.parameterized import bothmethod
from .core.data import Dataset
from .core.element import Element, Layout
from .core.layout import AdjointLayout
from .core.options import CallbackError, Store
from .core.overlay import NdOverlay, Overlay
from .core.spaces import GridSpace
from .streams import (
from .util import DynamicMap
@classmethod
def _build_selection_streams(cls, inst):
    style_stream = _Styles(colors=[inst.unselected_color, inst.selected_color], alpha=inst.unselected_alpha)
    cmap_streams = [_Cmap(cmap=inst.unselected_cmap), _Cmap(cmap=inst.selected_cmap)]

    def update_colors(*_):
        colors = [inst.unselected_color, inst.selected_color]
        style_stream.event(colors=colors, alpha=inst.unselected_alpha)
        cmap_streams[0].event(cmap=inst.unselected_cmap)
        if cmap_streams[1] is not None:
            cmap_streams[1].event(cmap=inst.selected_cmap)
    inst.param.watch(update_colors, ['unselected_color', 'selected_color', 'unselected_alpha'])
    exprs_stream = _SelectionExprLayers(inst._selection_override, inst._cross_filter_stream)
    return _SelectionStreams(style_stream=style_stream, exprs_stream=exprs_stream, cmap_streams=cmap_streams)