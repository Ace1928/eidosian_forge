from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
class timedelta_helper:
    """
    Helper for computing timedelta breaks
    and labels.

    How to use - breaks?

    1. Initialise with a timedelta sequence/limits.
    2. Get the scaled limits and use those to calculate
       breaks using a general purpose breaks calculating
       routine. The scaled limits are in numerical format.
    3. Convert the computed breaks from numeric into timedelta.

    See, :class:`breaks_timedelta`

    How to use - formating?

    1. Call :meth:`format_info` with the timedelta values to be
       formatted and get back a tuple of numeric values and
       the units for those values.
    2. Format the values with a general purpose formatting
       routing.

    See, :class:`~mizani.labels.label_timedelta`
    """
    x: NDArrayTimedelta | Sequence[Timedelta]
    units: DurationUnit
    limits: TupleFloat2
    package: Literal['pandas', 'cpython']
    factor: float

    def __init__(self, x: NDArrayTimedelta | Sequence[Timedelta], units: Optional[DurationUnit]=None):
        self.x = x
        self.package = self.determine_package(x[0])
        _limits = (min(x), max(x))
        self.limits = (self.value(_limits[0]), self.value(_limits[1]))
        self.units = units or self.best_units(_limits)
        self.factor = self.get_scaling_factor(self.units)

    @classmethod
    def determine_package(cls, td: Timedelta) -> Literal['pandas', 'cpython']:
        if hasattr(td, 'components'):
            package = 'pandas'
        elif hasattr(td, 'total_seconds'):
            package = 'cpython'
        else:
            msg = f'{td.__class__} format not yet supported.'
            raise ValueError(msg)
        return package

    @classmethod
    def format_info(cls, x: NDArrayTimedelta, units: Optional[DurationUnit]=None) -> tuple[NDArrayFloat, DurationUnit]:
        helper = cls(x, units)
        return (helper.timedelta_to_numeric(x), helper.units)

    def best_units(self, x: NDArrayTimedelta | Sequence[Timedelta]) -> DurationUnit:
        """
        Determine good units for representing a sequence of timedeltas
        """
        ts_range = self.value(max(x)) - self.value(min(x))
        package = self.determine_package(x[0])
        if package == 'pandas':
            cuts: list[tuple[float, DurationUnit]] = [(0.9, 'us'), (0.9, 'ms'), (0.9, 's'), (9, 'min'), (6, 'h'), (4, 'day'), (4, 'week'), (4, 'month'), (3, 'year')]
            denomination = NANOSECONDS
            base_units = 'ns'
        else:
            cuts = [(0.9, 's'), (9, 'min'), (6, 'h'), (4, 'day'), (4, 'week'), (4, 'month'), (3, 'year')]
            denomination = SECONDS
            base_units = 'ms'
        for size, units in reversed(cuts):
            if ts_range >= size * denomination[units]:
                return units
        return base_units

    @staticmethod
    def value(td: Timedelta) -> float:
        """
        Return the numeric value representation on a timedelta
        """
        if isinstance(td, pd.Timedelta):
            return td.value
        else:
            return td.total_seconds()

    def scaled_limits(self) -> TupleFloat2:
        """
        Minimum and Maximum to use for computing breaks
        """
        _min = self.limits[0] / self.factor
        _max = self.limits[1] / self.factor
        return (_min, _max)

    def timedelta_to_numeric(self, timedeltas: NDArrayTimedelta) -> NDArrayFloat:
        """
        Convert sequence of timedelta to numerics
        """
        return np.array([self.to_numeric(td) for td in timedeltas])

    def numeric_to_timedelta(self, values: NDArrayFloat) -> NDArrayTimedelta:
        """
        Convert sequence of numerical values to timedelta
        """
        if self.package == 'pandas':
            return np.array([pd.Timedelta(int(x * self.factor), unit='ns') for x in values])
        else:
            return np.array([timedelta(seconds=x * self.factor) for x in values])

    def get_scaling_factor(self, units):
        if self.package == 'pandas':
            return NANOSECONDS[units]
        else:
            return SECONDS[units]

    def to_numeric(self, td: Timedelta) -> float:
        """
        Convert timedelta to a number corresponding to the
        appropriate units. The appropriate units are those
        determined with the object is initialised.
        """
        if isinstance(td, pd.Timedelta):
            return td.value / NANOSECONDS[self.units]
        else:
            return td.total_seconds() / SECONDS[self.units]