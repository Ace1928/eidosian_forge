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
class IntervalProperty(Property):
    """A numeric property where scale range can be defined as an interval."""
    legend = True
    normed = True
    _default_range: tuple[float, float] = (0, 1)

    @property
    def default_range(self) -> tuple[float, float]:
        """Min and max values used by default for semantic mapping."""
        return self._default_range

    def _forward(self, values: ArrayLike) -> ArrayLike:
        """Transform applied to native values before linear mapping into interval."""
        return values

    def _inverse(self, values: ArrayLike) -> ArrayLike:
        """Transform applied to results of mapping that returns to native values."""
        return values

    def infer_scale(self, arg: Any, data: Series) -> Scale:
        """Given data and a scaling argument, initialize appropriate scale class."""
        var_type = variable_type(data, boolean_type='boolean', strict_boolean=True)
        if var_type == 'boolean':
            return Boolean(arg)
        elif isinstance(arg, (list, dict)):
            return Nominal(arg)
        elif var_type == 'categorical':
            return Nominal(arg)
        elif var_type == 'datetime':
            return Temporal(arg)
        else:
            return Continuous(arg)

    def get_mapping(self, scale: Scale, data: Series) -> Mapping:
        """Return a function that maps from data domain to property range."""
        if isinstance(scale, Nominal):
            return self._get_nominal_mapping(scale, data)
        elif isinstance(scale, Boolean):
            return self._get_boolean_mapping(scale, data)
        if scale.values is None:
            vmin, vmax = self._forward(self.default_range)
        elif isinstance(scale.values, tuple) and len(scale.values) == 2:
            vmin, vmax = self._forward(scale.values)
        else:
            if isinstance(scale.values, tuple):
                actual = f'{len(scale.values)}-tuple'
            else:
                actual = str(type(scale.values))
            scale_class = scale.__class__.__name__
            err = ' '.join([f'Values for {self.variable} variables with {scale_class} scale', f'must be 2-tuple; not {actual}.'])
            raise TypeError(err)

        def mapping(x):
            return self._inverse(np.multiply(x, vmax - vmin) + vmin)
        return mapping

    def _get_nominal_mapping(self, scale: Nominal, data: Series) -> Mapping:
        """Identify evenly-spaced values using interval or explicit mapping."""
        levels = categorical_order(data, scale.order)
        values = self._get_values(scale, levels)

        def mapping(x):
            ixs = np.asarray(x, np.intp)
            out = np.full(len(x), np.nan)
            use = np.isfinite(x)
            out[use] = np.take(values, ixs[use])
            return out
        return mapping

    def _get_boolean_mapping(self, scale: Boolean, data: Series) -> Mapping:
        """Identify evenly-spaced values using interval or explicit mapping."""
        values = self._get_values(scale, [True, False])

        def mapping(x):
            out = np.full(len(x), np.nan)
            use = np.isfinite(x)
            out[use] = np.where(x[use], *values)
            return out
        return mapping

    def _get_values(self, scale: Scale, levels: list) -> list:
        """Validate scale.values and identify a value for each level."""
        if isinstance(scale.values, dict):
            self._check_dict_entries(levels, scale.values)
            values = [scale.values[x] for x in levels]
        elif isinstance(scale.values, list):
            values = self._check_list_length(levels, scale.values)
        else:
            if scale.values is None:
                vmin, vmax = self.default_range
            elif isinstance(scale.values, tuple):
                vmin, vmax = scale.values
            else:
                scale_class = scale.__class__.__name__
                err = ' '.join([f'Values for {self.variable} variables with {scale_class} scale', f'must be a dict, list or tuple; not {type(scale.values)}'])
                raise TypeError(err)
            vmin, vmax = self._forward([vmin, vmax])
            values = list(self._inverse(np.linspace(vmax, vmin, len(levels))))
        return values