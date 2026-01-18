import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def _get_lasso_selection(self, xdim, ydim, geometry, **kwargs):
    from .path import Path
    bbox = {xdim.name: geometry[:, 0], ydim.name: geometry[:, 1]}
    expr = dim.pipe(spatial_poly_select, xdim, dim(ydim), geometry=geometry)
    index_cols = kwargs.get('index_cols')
    if index_cols:
        selection = self[expr.apply(self, expanded=False)]
        selection_expr = self._get_index_expr(index_cols, selection)
        return (selection_expr, bbox, None)
    return (expr, bbox, Path([np.concatenate([geometry, geometry[:1]])]))