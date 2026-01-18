from __future__ import annotations
import re
import typing
from warnings import warn
import numpy as np
import pandas as pd
from .._utils import join_keys, match
from ..exceptions import PlotnineError, PlotnineWarning
from .facet import (
from .strips import Strips, strip
class facet_wrap(facet):
    """
    Wrap 1D Panels onto 2D surface

    Parameters
    ----------
    facets :
        Variables to groupby and plot on different panels.
        If a string formula is used it should be right sided,
        e.g `"~ a + b"`, `("a", "b")`
    nrow : int, default=None
        Number of rows
    ncol : int, default=None
        Number of columns
    scales :
        Whether `x` or `y` scales should be allowed (free)
        to vary according to the data on each of the panel.
    shrink :
        Whether to shrink the scales to the output of the
        statistics instead of the raw data.
    labeller :
        How to label the facets. A string value if it should be
        one of `["label_value", "label_both", "label_context"]`{.py}.
    as_table :
        If `True`, the facets are laid out like a table with
        the highest values at the bottom-right. If `False`
        the facets are laid out like a plot with the highest
        value a the top-right
    drop :
        If `True`, all factor levels not used in the data
        will automatically be dropped. If `False`, all
        factor levels will be shown, regardless of whether
        or not they appear in the data.
    dir :
        Direction in which to layout the panels. `h` for
        horizontal and `v` for vertical.
    """

    def __init__(self, facets: Optional[str | Sequence[str]]=None, *, nrow: Optional[int]=None, ncol: Optional[int]=None, scales: Literal['fixed', 'free', 'free_x', 'free_y']='fixed', shrink: bool=True, labeller: Literal['label_value', 'label_both', 'label_context']='label_value', as_table: bool=True, drop: bool=True, dir: Literal['h', 'v']='h'):
        super().__init__(scales=scales, shrink=shrink, labeller=labeller, as_table=as_table, drop=drop, dir=dir)
        self.vars = parse_wrap_facets(facets)
        self._nrow, self._ncol = check_dimensions(nrow, ncol)

    def compute_layout(self, data: list[pd.DataFrame]) -> pd.DataFrame:
        if not self.vars:
            self.nrow, self.ncol = (1, 1)
            return layout_null()
        base = combine_vars(data, self.environment, self.vars, drop=self.drop)
        n = len(base)
        dims = wrap_dims(n, self._nrow, self._ncol)
        _id = np.arange(1, n + 1)
        if self.dir == 'v':
            dims = dims[::-1]
        if self.as_table:
            row = (_id - 1) // dims[1] + 1
        else:
            row = dims[0] - (_id - 1) // dims[1]
        col = (_id - 1) % dims[1] + 1
        layout = pd.DataFrame({'PANEL': pd.Categorical(range(1, n + 1)), 'ROW': row.astype(int), 'COL': col.astype(int)})
        if self.dir == 'v':
            layout.rename(columns={'ROW': 'COL', 'COL': 'ROW'}, inplace=True)
        layout = pd.concat([layout, base], axis=1)
        self.nrow = layout['ROW'].nunique()
        self.ncol = layout['COL'].nunique()
        n = layout.shape[0]
        layout['SCALE_X'] = range(1, n + 1) if self.free['x'] else 1
        layout['SCALE_Y'] = range(1, n + 1) if self.free['y'] else 1
        x_idx = [df['ROW'].idxmax() for _, df in layout.groupby('COL')]
        y_idx = [df['COL'].idxmin() for _, df in layout.groupby('ROW')]
        layout['AXIS_X'] = False
        layout['AXIS_Y'] = False
        _loc = layout.columns.get_loc
        layout.iloc[x_idx, _loc('AXIS_X')] = True
        layout.iloc[y_idx, _loc('AXIS_Y')] = True
        if self.free['x']:
            layout.loc[:, 'AXIS_X'] = True
        if self.free['y']:
            layout.loc[:, 'AXIS_Y'] = True
        return layout

    def map(self, data: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
        if not len(data):
            data['PANEL'] = pd.Categorical([], categories=layout['PANEL'].cat.categories, ordered=True)
            return data
        facet_vals = eval_facet_vars(data, self.vars, self.environment)
        data, facet_vals = add_missing_facets(data, layout, self.vars, facet_vals)
        if len(facet_vals) and len(facet_vals.columns):
            keys = join_keys(facet_vals, layout, self.vars)
            data['PANEL'] = match(keys['x'], keys['y'], start=1)
        else:
            data['PANEL'] = 1
        data['PANEL'] = pd.Categorical(data['PANEL'], categories=layout['PANEL'].cat.categories, ordered=True)
        data.reset_index(drop=True, inplace=True)
        return data

    def make_strips(self, layout_info: layout_details, ax: Axes) -> Strips:
        if not self.vars:
            return Strips([])
        s = strip(self.vars, layout_info, self, ax, 'top')
        return Strips([s])