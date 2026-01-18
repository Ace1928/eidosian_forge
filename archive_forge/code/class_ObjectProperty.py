from __future__ import annotations
import itertools
import warnings
import numpy as np
from numpy.typing import ArrayLike
from pandas import Series
import matplotlib as mpl
from matplotlib.colors import to_rgb, to_rgba, to_rgba_array
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from seaborn._core.scales import Scale, Boolean, Continuous, Nominal, Temporal
from seaborn._core.rules import categorical_order, variable_type
from seaborn.palettes import QUAL_PALETTES, color_palette, blend_palette
from seaborn.utils import get_color_cycle
from typing import Any, Callable, Tuple, List, Union, Optional
class ObjectProperty(Property):
    """A property defined by arbitrary an object, with inherently nominal scaling."""
    legend = True
    normed = False
    null_value: Any = None

    def _default_values(self, n: int) -> list:
        raise NotImplementedError()

    def default_scale(self, data: Series) -> Scale:
        var_type = variable_type(data, boolean_type='boolean', strict_boolean=True)
        return Boolean() if var_type == 'boolean' else Nominal()

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        var_type = variable_type(data, boolean_type='boolean', strict_boolean=True)
        return Boolean(arg) if var_type == 'boolean' else Nominal(arg)

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """Define mapping as lookup into list of object values."""
        boolean_scale = isinstance(scale, Boolean)
        order = getattr(scale, 'order', [True, False] if boolean_scale else None)
        levels = categorical_order(data, order)
        values = self._get_values(scale, levels)
        if boolean_scale:
            values = values[::-1]

        def mapping(x):
            ixs = np.asarray(np.nan_to_num(x), np.intp)
            return [values[ix] if np.isfinite(x_i) else self.null_value for x_i, ix in zip(x, ixs)]
        return mapping

    def _get_values(self, scale: Scale, levels: list) -> list:
        """Validate scale.values and identify a value for each level."""
        n = len(levels)
        if isinstance(scale.values, dict):
            self._check_dict_entries(levels, scale.values)
            values = [scale.values[x] for x in levels]
        elif isinstance(scale.values, list):
            values = self._check_list_length(levels, scale.values)
        elif scale.values is None:
            values = self._default_values(n)
        else:
            msg = ' '.join([f'Scale values for a {self.variable} variable must be provided', f'in a dict or list; not {type(scale.values)}.'])
            raise TypeError(msg)
        values = [self.standardize(x) for x in values]
        return values