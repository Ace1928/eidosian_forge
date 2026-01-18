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
@param.depends('selection_expr', watch=True)
def _update_pipes(self):
    sel_expr = self.selection_expr
    for pipe, ds, raw in self._datasets:
        ref = ds._plot_id
        self._cache[ref] = ds_cache = self._cache.get(ref, {})
        if sel_expr in ds_cache:
            data = ds_cache[sel_expr]
            return pipe.event(data=data.data)
        else:
            ds_cache.clear()
        sel_ds = SelectionDisplay._select(ds, sel_expr, self._cache)
        ds_cache[sel_expr] = sel_ds
        pipe.event(data=sel_ds.data if raw else sel_ds)