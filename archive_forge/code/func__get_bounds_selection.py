import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def _get_bounds_selection(self, xdim, ydim, **kwargs):
    from .geom import Rectangles
    (x0, x1), _, (y0, y1), _ = self._get_selection(**kwargs)
    bbox = {xdim.name: (x0, x1), ydim.name: (y0, y1)}
    index_cols = kwargs.get('index_cols')
    expr = dim.pipe(spatial_bounds_select, xdim, dim(ydim), bounds=(x0, y0, x1, y1))
    if index_cols:
        selection = self[expr.apply(self, expanded=False)]
        selection_expr = self._get_index_expr(index_cols, selection)
        return (selection_expr, bbox, None)
    return (expr, bbox, Rectangles([(x0, y0, x1, y1)]))