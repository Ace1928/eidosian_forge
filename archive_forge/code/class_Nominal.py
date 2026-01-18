from __future__ import annotations
import re
from copy import copy
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Tuple, Optional, ClassVar
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import (
from matplotlib.dates import (
from matplotlib.axis import Axis
from matplotlib.scale import ScaleBase
from pandas import Series
from seaborn._core.rules import categorical_order
from seaborn._core.typing import Default, default
from typing import TYPE_CHECKING
@dataclass
class Nominal(Scale):
    """
    A categorical scale without relative importance / magnitude.
    """
    values: tuple | str | list | dict | None = None
    order: list | None = None
    _priority: ClassVar[int] = 4

    def _setup(self, data: Series, prop: Property, axis: Axis | None=None) -> Scale:
        new = copy(self)
        if new._tick_params is None:
            new = new.tick()
        if new._label_params is None:
            new = new.label()
        stringify = np.vectorize(format, otypes=['object'])
        units_seed = categorical_order(data, new.order)

        class CatScale(mpl.scale.LinearScale):

            def set_default_locators_and_formatters(self, axis):
                ...
        mpl_scale = CatScale(data.name)
        if axis is None:
            axis = PseudoAxis(mpl_scale)
            axis.set_view_interval(0, len(units_seed) - 1)
        new._matplotlib_scale = mpl_scale
        axis.update_units(stringify(np.array(units_seed)))

        def convert_units(x):
            keep = np.array([x_ in units_seed for x_ in x], bool)
            out = np.full(len(x), np.nan)
            out[keep] = axis.convert_units(stringify(x[keep]))
            return out
        new._pipeline = [convert_units, prop.get_mapping(new, data)]
        new._spacer = _default_spacer
        if prop.legend:
            new._legend = (units_seed, list(stringify(units_seed)))
        return new

    def _finalize(self, p: Plot, axis: Axis) -> None:
        ax = axis.axes
        name = axis.axis_name
        axis.grid(False, which='both')
        if name not in p._limits:
            nticks = len(axis.get_major_ticks())
            lo, hi = (-0.5, nticks - 0.5)
            if name == 'y':
                lo, hi = (hi, lo)
            set_lim = getattr(ax, f'set_{name}lim')
            set_lim(lo, hi, auto=None)

    def tick(self, locator: Locator | None=None) -> Nominal:
        """
        Configure the selection of ticks for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.
            At the moment, it is probably not very useful.

        Parameters
        ----------
        locator : :class:`matplotlib.ticker.Locator` subclass
            Pre-configured matplotlib locator; other parameters will not be used.

        Returns
        -------
        Copy of self with new tick configuration.

        """
        new = copy(self)
        new._tick_params = {'locator': locator}
        return new

    def label(self, formatter: Formatter | None=None) -> Nominal:
        """
        Configure the selection of labels for the scale's axis or legend.

        .. note::
            This API is under construction and will be enhanced over time.
            At the moment, it is probably not very useful.

        Parameters
        ----------
        formatter : :class:`matplotlib.ticker.Formatter` subclass
            Pre-configured matplotlib formatter; other parameters will not be used.

        Returns
        -------
        scale
            Copy of self with new tick configuration.

        """
        new = copy(self)
        new._label_params = {'formatter': formatter}
        return new

    def _get_locators(self, locator):
        if locator is not None:
            return (locator, None)
        locator = mpl.category.StrCategoryLocator({})
        return (locator, None)

    def _get_formatter(self, locator, formatter):
        if formatter is not None:
            return formatter
        formatter = mpl.category.StrCategoryFormatter({})
        return formatter