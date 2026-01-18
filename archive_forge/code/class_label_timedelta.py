from __future__ import annotations
import re
import typing
from bisect import bisect_right
from dataclasses import dataclass
from zoneinfo import ZoneInfo
import numpy as np
from .breaks import timedelta_helper
from .utils import (
@dataclass
class label_timedelta:
    """
    Timedelta labels

    Parameters
    ----------
    units : str, optional
        The units in which the breaks will be computed.
        If None, they are decided automatically. Otherwise,
        the value should be one of::

            'ns'    # nanoseconds
            'us'    # microseconds
            'ms'    # milliseconds
            's'     # seconds
            'min'   # minute
            'h'     # hour
            'day'     # day
            'week'  # week
            'month' # month
            'year'  # year

    show_units : bool
        Whether to append the units symbol to the values.
    zero_has_units : bool
        If True a value of zero
    usetex : bool
        If True, they microseconds identifier string is
        rendered with greek letter *mu*. Default is False.
    space : bool
        If True add a space between the value and the units
    use_plurals : bool
        If True, for the when the value is not 1 and the units are
        one of `week`, `month` and `year`, the plural form of the
        unit is used e.g. `2 weeks`.

    Examples
    --------
    >>> from datetime import timedelta
    >>> x = [timedelta(days=31*i) for i in range(5)]
    >>> label_timedelta()(x)
    ['0 months', '1 month', '2 months', '3 months', '4 months']
    >>> label_timedelta(use_plurals=False)(x)
    ['0 month', '1 month', '2 month', '3 month', '4 month']
    >>> label_timedelta(units='day')(x)
    ['0 days', '31 days', '62 days', '93 days', '124 days']
    >>> label_timedelta(units='day', zero_has_units=False)(x)
    ['0', '31 days', '62 days', '93 days', '124 days']
    >>> label_timedelta(units='day', show_units=False)(x)
    ['0', '31', '62', '93', '124']
    """
    units: Optional[DurationUnit] = None
    show_units: bool = True
    zero_has_units: bool = True
    usetex: bool = False
    space: bool = True
    use_plurals: bool = True

    def __call__(self, x: NDArrayTimedelta) -> Sequence[str]:
        if len(x) == 0:
            return []
        values, units = timedelta_helper.format_info(x, self.units)
        labels = list(label_number()(values))
        if self.show_units:
            if self.usetex and units == 'us':
                units = '$\\mu s$'
            if self.use_plurals and units in ('day', 'week', 'month', 'year'):
                units_plural = f'{units}s'
            else:
                units_plural = units
            if self.space:
                units = f' {units}'
                units_plural = f' {units_plural}'
            for i, (num, label) in enumerate(zip(values, labels)):
                if num == 0 and (not self.zero_has_units):
                    continue
                elif num == 1:
                    labels[i] = f'{label}{units}'
                else:
                    labels[i] = f'{label}{units_plural}'
        return labels