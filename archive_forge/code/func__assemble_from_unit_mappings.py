from __future__ import annotations
from collections import abc
from datetime import date
from functools import partial
from itertools import islice
from typing import (
import warnings
import numpy as np
from pandas._libs import (
from pandas._libs.tslibs import (
from pandas._libs.tslibs.conversion import cast_from_unit_vectorized
from pandas._libs.tslibs.parsing import (
from pandas._libs.tslibs.strptime import array_strptime
from pandas._typing import (
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.arrays import (
from pandas.core.algorithms import unique
from pandas.core.arrays import ArrowExtensionArray
from pandas.core.arrays.base import ExtensionArray
from pandas.core.arrays.datetimes import (
from pandas.core.construction import extract_array
from pandas.core.indexes.base import Index
from pandas.core.indexes.datetimes import DatetimeIndex
def _assemble_from_unit_mappings(arg, errors: DateTimeErrorChoices, utc: bool):
    """
    assemble the unit specified fields from the arg (DataFrame)
    Return a Series for actual parsing

    Parameters
    ----------
    arg : DataFrame
    errors : {'ignore', 'raise', 'coerce'}, default 'raise'

        - If :const:`'raise'`, then invalid parsing will raise an exception
        - If :const:`'coerce'`, then invalid parsing will be set as :const:`NaT`
        - If :const:`'ignore'`, then invalid parsing will return the input
    utc : bool
        Whether to convert/localize timestamps to UTC.

    Returns
    -------
    Series
    """
    from pandas import DataFrame, to_numeric, to_timedelta
    arg = DataFrame(arg)
    if not arg.columns.is_unique:
        raise ValueError('cannot assemble with duplicate keys')

    def f(value):
        if value in _unit_map:
            return _unit_map[value]
        if value.lower() in _unit_map:
            return _unit_map[value.lower()]
        return value
    unit = {k: f(k) for k in arg.keys()}
    unit_rev = {v: k for k, v in unit.items()}
    required = ['year', 'month', 'day']
    req = sorted(set(required) - set(unit_rev.keys()))
    if len(req):
        _required = ','.join(req)
        raise ValueError(f'to assemble mappings requires at least that [year, month, day] be specified: [{_required}] is missing')
    excess = sorted(set(unit_rev.keys()) - set(_unit_map.values()))
    if len(excess):
        _excess = ','.join(excess)
        raise ValueError(f'extra keys have been passed to the datetime assemblage: [{_excess}]')

    def coerce(values):
        values = to_numeric(values, errors=errors)
        if is_integer_dtype(values.dtype):
            values = values.astype('int64', copy=False)
        return values
    values = coerce(arg[unit_rev['year']]) * 10000 + coerce(arg[unit_rev['month']]) * 100 + coerce(arg[unit_rev['day']])
    try:
        values = to_datetime(values, format='%Y%m%d', errors=errors, utc=utc)
    except (TypeError, ValueError) as err:
        raise ValueError(f'cannot assemble the datetimes: {err}') from err
    units: list[UnitChoices] = ['h', 'm', 's', 'ms', 'us', 'ns']
    for u in units:
        value = unit_rev.get(u)
        if value is not None and value in arg:
            try:
                values += to_timedelta(coerce(arg[value]), unit=u, errors=errors)
            except (TypeError, ValueError) as err:
                raise ValueError(f'cannot assemble the datetimes [{value}]: {err}') from err
    return values