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
def _add_annotation_like(self, prop_singular, prop_plural, new_obj, row=None, col=None, secondary_y=None, exclude_empty_subplots=False):
    if row is not None and col is None:
        raise ValueError('Received row parameter but not col.\nrow and col must be specified together')
    elif col is not None and row is None:
        raise ValueError('Received col parameter but not row.\nrow and col must be specified together')
    if row is not None and _is_select_subplot_coordinates_arg(row, col):
        rows_cols = self._select_subplot_coordinates(row, col)
        for r, c in rows_cols:
            self._add_annotation_like(prop_singular, prop_plural, new_obj, row=r, col=c, secondary_y=secondary_y, exclude_empty_subplots=exclude_empty_subplots)
        return self
    if row is not None:
        grid_ref = self._validate_get_grid_ref()
        if row > len(grid_ref):
            raise IndexError('row index %d out-of-bounds, row index must be between 1 and %d, inclusive.' % (row, len(grid_ref)))
        if col > len(grid_ref[row - 1]):
            raise IndexError('column index %d out-of-bounds, column index must be between 1 and %d, inclusive.' % (row, len(grid_ref[row - 1])))
        refs = grid_ref[row - 1][col - 1]
        if not refs:
            raise ValueError('No subplot found at position ({r}, {c})'.format(r=row, c=col))
        if refs[0].subplot_type != 'xy':
            raise ValueError('\nCannot add {prop_singular} to subplot at position ({r}, {c}) because subplot\nis of type {subplot_type}.'.format(prop_singular=prop_singular, r=row, c=col, subplot_type=refs[0].subplot_type))
        if new_obj.yref is None or new_obj.yref == 'y' or 'paper' in new_obj.yref or ('domain' in new_obj.yref):
            if len(refs) == 1 and secondary_y:
                raise ValueError('\n    Cannot add {prop_singular} to secondary y-axis of subplot at position ({r}, {c})\n    because subplot does not have a secondary y-axis'.format(prop_singular=prop_singular, r=row, c=col))
            if secondary_y:
                xaxis, yaxis = refs[1].layout_keys
            else:
                xaxis, yaxis = refs[0].layout_keys
            xref, yref = (xaxis.replace('axis', ''), yaxis.replace('axis', ''))
        else:
            yref = new_obj.yref
            xaxis = refs[0].layout_keys[0]
            xref = xaxis.replace('axis', '')
        if exclude_empty_subplots and (not self._subplot_not_empty(xref, yref, selector=bool(exclude_empty_subplots))):
            return self

        def _add_domain(ax_letter, new_axref):
            axref = ax_letter + 'ref'
            if axref in new_obj._props.keys() and 'domain' in new_obj[axref]:
                new_axref += ' domain'
            return new_axref
        xref, yref = map(lambda t: _add_domain(*t), zip(['x', 'y'], [xref, yref]))
        new_obj.update(xref=xref, yref=yref)
    self.layout[prop_plural] += (new_obj,)
    new_obj.update(xref=None, yref=None)
    return self