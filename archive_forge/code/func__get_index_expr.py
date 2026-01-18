import sys
from importlib.util import find_spec
import numpy as np
import pandas as pd
from ..core import Dataset, NdOverlay, util
from ..streams import Lasso, Selection1D, SelectionXY
from ..util.transform import dim
from .annotation import HSpan, VSpan
def _get_index_expr(self, index_cols, sel):
    if len(index_cols) == 1:
        index_dim = index_cols[0]
        vals = dim(index_dim).apply(sel, expanded=False, flat=True)
        expr = dim(index_dim).isin(list(util.unique_iterator(vals)))
    else:
        get_shape = dim(self.dataset.get_dimension(), np.shape)
        index_cols = [dim(self.dataset.get_dimension(c), np.ravel) for c in index_cols]
        vals = dim(index_cols[0], util.unique_zip, *index_cols[1:]).apply(sel, expanded=True, flat=True)
        contains = dim(index_cols[0], util.lzip, *index_cols[1:]).isin(vals, object=True)
        expr = dim(contains, np.reshape, get_shape)
    return expr