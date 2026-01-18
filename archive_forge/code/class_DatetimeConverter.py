from __future__ import annotations
import contextlib
import datetime as pydt
from datetime import (
import functools
from typing import (
import warnings
import matplotlib.dates as mdates
from matplotlib.ticker import (
from matplotlib.transforms import nonsingular
import matplotlib.units as munits
import numpy as np
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import (
from pandas._typing import (
from pandas.core.dtypes.common import (
from pandas import (
import pandas.core.common as com
from pandas.core.indexes.datetimes import date_range
from pandas.core.indexes.period import (
import pandas.core.tools.datetimes as tools
class DatetimeConverter(mdates.DateConverter):

    @staticmethod
    def convert(values, unit, axis):
        if is_nested_list_like(values):
            values = [DatetimeConverter._convert_1d(v, unit, axis) for v in values]
        else:
            values = DatetimeConverter._convert_1d(values, unit, axis)
        return values

    @staticmethod
    def _convert_1d(values, unit, axis):

        def try_parse(values):
            try:
                return mdates.date2num(tools.to_datetime(values))
            except Exception:
                return values
        if isinstance(values, (datetime, pydt.date, np.datetime64, pydt.time)):
            return mdates.date2num(values)
        elif is_integer(values) or is_float(values):
            return values
        elif isinstance(values, str):
            return try_parse(values)
        elif isinstance(values, (list, tuple, np.ndarray, Index, Series)):
            if isinstance(values, Series):
                values = Index(values)
            if isinstance(values, Index):
                values = values.values
            if not isinstance(values, np.ndarray):
                values = com.asarray_tuplesafe(values)
            if is_integer_dtype(values) or is_float_dtype(values):
                return values
            try:
                values = tools.to_datetime(values)
            except Exception:
                pass
            values = mdates.date2num(values)
        return values

    @staticmethod
    def axisinfo(unit: tzinfo | None, axis) -> munits.AxisInfo:
        """
        Return the :class:`~matplotlib.units.AxisInfo` for *unit*.

        *unit* is a tzinfo instance or None.
        The *axis* argument is required but not used.
        """
        tz = unit
        majloc = PandasAutoDateLocator(tz=tz)
        majfmt = PandasAutoDateFormatter(majloc, tz=tz)
        datemin = pydt.date(2000, 1, 1)
        datemax = pydt.date(2010, 1, 1)
        return munits.AxisInfo(majloc=majloc, majfmt=majfmt, label='', default_limits=(datemin, datemax))