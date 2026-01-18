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
class ColorListSelectionDisplay(SelectionDisplay):
    """
    Selection display class for elements that support coloring by a
    vectorized color list.
    """

    def __init__(self, color_prop='color', alpha_prop='alpha', backend=None):
        self.color_props = [color_prop]
        self.alpha_props = [alpha_prop]
        self.backend = backend

    def build_selection(self, selection_streams, hvobj, operations, region_stream=None, cache=None):
        if cache is None:
            cache = {}

        def _build_selection(el, colors, alpha, exprs, **kwargs):
            from .plotting.util import linear_gradient
            ds = el.dataset
            selection_exprs = exprs[1:]
            unselected_color = colors[0]
            unselected_color = unselected_color or '#e6e9ec'
            backup_clr = linear_gradient(unselected_color, '#000000', 7)[2]
            selected_colors = [c or backup_clr for c in colors[1:]]
            n = len(ds)
            clrs = np.array([unselected_color] + list(selected_colors))
            color_inds = np.zeros(n, dtype='int8')
            for i, expr in zip(range(1, len(clrs)), selection_exprs):
                if not expr:
                    color_inds[:] = i
                else:
                    color_inds[expr.apply(ds)] = i
            colors = clrs[color_inds]
            color_opts = {color_prop: colors for color_prop in self.color_props}
            return el.pipeline(ds).opts(backend=self.backend, clone=True, **color_opts)
        sel_streams = [selection_streams.style_stream, selection_streams.exprs_stream]
        hvobj = hvobj.apply(_build_selection, streams=sel_streams, per_element=True, cache=cache)
        for op in operations:
            hvobj = op(hvobj)
        return hvobj