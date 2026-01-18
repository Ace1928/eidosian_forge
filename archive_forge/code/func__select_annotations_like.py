import collections
from collections import OrderedDict
import re
import warnings
from contextlib import contextmanager
from copy import deepcopy, copy
import itertools
from functools import reduce
from _plotly_utils.utils import (
from _plotly_utils.exceptions import PlotlyKeyError
from .optional_imports import get_module
from . import shapeannotation
from . import _subplots
def _select_annotations_like(self, prop, selector=None, row=None, col=None, secondary_y=None):
    """
        Helper to select annotation-like elements from a layout object array.
        Compatible with layout.annotations, layout.shapes, and layout.images
        """
    xref_to_col = {}
    yref_to_row = {}
    yref_to_secondary_y = {}
    if isinstance(row, int) or isinstance(col, int) or secondary_y is not None:
        grid_ref = self._validate_get_grid_ref()
        for r, subplot_row in enumerate(grid_ref):
            for c, subplot_refs in enumerate(subplot_row):
                if not subplot_refs:
                    continue
                for i, subplot_ref in enumerate(subplot_refs):
                    if subplot_ref.subplot_type == 'xy':
                        is_secondary_y = i == 1
                        xaxis, yaxis = subplot_ref.layout_keys
                        xref = xaxis.replace('axis', '')
                        yref = yaxis.replace('axis', '')
                        xref_to_col[xref] = c + 1
                        yref_to_row[yref] = r + 1
                        yref_to_secondary_y[yref] = is_secondary_y

    def _filter_row(obj):
        """Filter objects in rows by column"""
        return col is None or xref_to_col.get(obj.xref, None) == col

    def _filter_col(obj):
        """Filter objects in columns by row"""
        return row is None or yref_to_row.get(obj.yref, None) == row

    def _filter_sec_y(obj):
        """Filter objects on secondary y axes"""
        return secondary_y is None or yref_to_secondary_y.get(obj.yref, None) == secondary_y
    funcs = [_filter_row, _filter_col, _filter_sec_y]
    return _generator(self._filter_by_selector(self.layout[prop], funcs, selector))