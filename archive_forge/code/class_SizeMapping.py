from __future__ import annotations
import warnings
import itertools
from copy import copy
from collections import UserString
from collections.abc import Iterable, Sequence, Mapping
from numbers import Number
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib as mpl
from seaborn._core.data import PlotData
from seaborn.palettes import (
from seaborn.utils import (
class SizeMapping(SemanticMapping):
    """Mapping that sets artist sizes according to data values."""
    norm = None

    def __init__(self, plotter, sizes=None, order=None, norm=None):
        """Map the levels of the `size` variable to distinct values.

        Parameters
        ----------
        # TODO add generic parameters

        """
        super().__init__(plotter)
        data = plotter.plot_data.get('size', pd.Series(dtype=float))
        if data.notna().any():
            map_type = self.infer_map_type(norm, sizes, plotter.var_types['size'])
            if map_type == 'numeric':
                levels, lookup_table, norm, size_range = self.numeric_mapping(data, sizes, norm)
            elif map_type == 'categorical':
                levels, lookup_table = self.categorical_mapping(data, sizes, order)
                size_range = None
            else:
                levels, lookup_table = self.categorical_mapping(list(data), sizes, order)
                size_range = None
            self.map_type = map_type
            self.levels = levels
            self.norm = norm
            self.sizes = sizes
            self.size_range = size_range
            self.lookup_table = lookup_table

    def infer_map_type(self, norm, sizes, var_type):
        if norm is not None:
            map_type = 'numeric'
        elif isinstance(sizes, (dict, list)):
            map_type = 'categorical'
        else:
            map_type = var_type
        return map_type

    def _lookup_single(self, key):
        try:
            value = self.lookup_table[key]
        except KeyError:
            normed = self.norm(key)
            if np.ma.is_masked(normed):
                normed = np.nan
            value = self.size_range[0] + normed * np.ptp(self.size_range)
        return value

    def categorical_mapping(self, data, sizes, order):
        levels = categorical_order(data, order)
        if isinstance(sizes, dict):
            missing = set(levels) - set(sizes)
            if any(missing):
                err = f'Missing sizes for the following levels: {missing}'
                raise ValueError(err)
            lookup_table = sizes.copy()
        elif isinstance(sizes, list):
            sizes = self._check_list_length(levels, sizes, 'sizes')
            lookup_table = dict(zip(levels, sizes))
        else:
            if isinstance(sizes, tuple):
                if len(sizes) != 2:
                    err = 'A `sizes` tuple must have only 2 values'
                    raise ValueError(err)
            elif sizes is not None:
                err = f'Value for `sizes` not understood: {sizes}'
                raise ValueError(err)
            else:
                sizes = self.plotter._default_size_range
            sizes = np.linspace(*sizes, len(levels))[::-1]
            lookup_table = dict(zip(levels, sizes))
        return (levels, lookup_table)

    def numeric_mapping(self, data, sizes, norm):
        if isinstance(sizes, dict):
            levels = list(np.sort(list(sizes)))
            size_values = sizes.values()
            size_range = (min(size_values), max(size_values))
        else:
            levels = list(np.sort(remove_na(data.unique())))
            if isinstance(sizes, tuple):
                if len(sizes) != 2:
                    err = 'A `sizes` tuple must have only 2 values'
                    raise ValueError(err)
                size_range = sizes
            elif sizes is not None:
                err = f'Value for `sizes` not understood: {sizes}'
                raise ValueError(err)
            else:
                size_range = self.plotter._default_size_range
        if norm is None:
            norm = mpl.colors.Normalize()
        elif isinstance(norm, tuple):
            norm = mpl.colors.Normalize(*norm)
        elif not isinstance(norm, mpl.colors.Normalize):
            err = f'Value for size `norm` parameter not understood: {norm}'
            raise ValueError(err)
        else:
            norm = copy(norm)
        norm.clip = True
        if not norm.scaled():
            norm(levels)
        sizes_scaled = norm(levels)
        if isinstance(sizes, dict):
            lookup_table = sizes
        else:
            lo, hi = size_range
            sizes = lo + sizes_scaled * (hi - lo)
            lookup_table = dict(zip(levels, sizes))
        return (levels, lookup_table, norm, size_range)