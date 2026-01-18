from __future__ import annotations
import re
from typing import TYPE_CHECKING
import numpy as np
import pandas
from modin.logging import ClassLogger
from modin.utils import _inherit_docstrings
@_inherit_docstrings(pandas.core.indexes.accessors.CombinedDatetimelikeProperties)
class DatetimeProperties(ClassLogger):
    _series: Series
    _query_compiler: BaseQueryCompiler

    def __init__(self, data: Series):
        self._series = data
        self._query_compiler = data._query_compiler

    @pandas.util.cache_readonly
    def _Series(self):
        from .series import Series
        return Series

    @property
    def date(self):
        return self._Series(query_compiler=self._query_compiler.dt_date())

    @property
    def time(self):
        return self._Series(query_compiler=self._query_compiler.dt_time())

    @property
    def timetz(self):
        return self._Series(query_compiler=self._query_compiler.dt_timetz())

    @property
    def year(self):
        return self._Series(query_compiler=self._query_compiler.dt_year())

    @property
    def month(self):
        return self._Series(query_compiler=self._query_compiler.dt_month())

    @property
    def day(self):
        return self._Series(query_compiler=self._query_compiler.dt_day())

    @property
    def hour(self):
        return self._Series(query_compiler=self._query_compiler.dt_hour())

    @property
    def minute(self):
        return self._Series(query_compiler=self._query_compiler.dt_minute())

    @property
    def second(self):
        return self._Series(query_compiler=self._query_compiler.dt_second())

    @property
    def microsecond(self):
        return self._Series(query_compiler=self._query_compiler.dt_microsecond())

    @property
    def nanosecond(self):
        return self._Series(query_compiler=self._query_compiler.dt_nanosecond())

    @property
    def dayofweek(self):
        return self._Series(query_compiler=self._query_compiler.dt_dayofweek())
    day_of_week = dayofweek

    @property
    def weekday(self):
        return self._Series(query_compiler=self._query_compiler.dt_weekday())

    @property
    def dayofyear(self):
        return self._Series(query_compiler=self._query_compiler.dt_dayofyear())
    day_of_year = dayofyear

    @property
    def quarter(self):
        return self._Series(query_compiler=self._query_compiler.dt_quarter())

    @property
    def is_month_start(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_month_start())

    @property
    def is_month_end(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_month_end())

    @property
    def is_quarter_start(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_quarter_start())

    @property
    def is_quarter_end(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_quarter_end())

    @property
    def is_year_start(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_year_start())

    @property
    def is_year_end(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_year_end())

    @property
    def is_leap_year(self):
        return self._Series(query_compiler=self._query_compiler.dt_is_leap_year())

    @property
    def daysinmonth(self):
        return self._Series(query_compiler=self._query_compiler.dt_daysinmonth())

    @property
    def days_in_month(self):
        return self._Series(query_compiler=self._query_compiler.dt_days_in_month())

    @property
    def tz(self) -> 'tzinfo | None':
        dtype = self._series.dtype
        if isinstance(dtype, np.dtype):
            return None
        return dtype.tz

    @property
    def freq(self):
        return self._query_compiler.dt_freq().to_pandas().squeeze()

    @property
    def unit(self):
        return self._Series(query_compiler=self._query_compiler.dt_unit()).iloc[0]

    def as_unit(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_as_unit(*args, **kwargs))

    def to_period(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_to_period(*args, **kwargs))

    def asfreq(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_asfreq(*args, **kwargs))

    def to_pydatetime(self):
        return self._Series(query_compiler=self._query_compiler.dt_to_pydatetime()).to_numpy()

    def tz_localize(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_tz_localize(*args, **kwargs))

    def tz_convert(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_tz_convert(*args, **kwargs))

    def normalize(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_normalize(*args, **kwargs))

    def strftime(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_strftime(*args, **kwargs))

    def round(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_round(*args, **kwargs))

    def floor(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_floor(*args, **kwargs))

    def ceil(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_ceil(*args, **kwargs))

    def month_name(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_month_name(*args, **kwargs))

    def day_name(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_day_name(*args, **kwargs))

    def total_seconds(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_total_seconds(*args, **kwargs))

    def to_pytimedelta(self) -> 'npt.NDArray[np.object_]':
        res = self._query_compiler.dt_to_pytimedelta()
        return res.to_numpy()[:, 0]

    @property
    def seconds(self):
        return self._Series(query_compiler=self._query_compiler.dt_seconds())

    @property
    def days(self):
        return self._Series(query_compiler=self._query_compiler.dt_days())

    @property
    def microseconds(self):
        return self._Series(query_compiler=self._query_compiler.dt_microseconds())

    @property
    def nanoseconds(self):
        return self._Series(query_compiler=self._query_compiler.dt_nanoseconds())

    @property
    def components(self):
        from .dataframe import DataFrame
        return DataFrame(query_compiler=self._query_compiler.dt_components())

    def isocalendar(self):
        from .dataframe import DataFrame
        return DataFrame(query_compiler=self._query_compiler.dt_isocalendar())

    @property
    def qyear(self):
        return self._Series(query_compiler=self._query_compiler.dt_qyear())

    @property
    def start_time(self):
        return self._Series(query_compiler=self._query_compiler.dt_start_time())

    @property
    def end_time(self):
        return self._Series(query_compiler=self._query_compiler.dt_end_time())

    def to_timestamp(self, *args, **kwargs):
        return self._Series(query_compiler=self._query_compiler.dt_to_timestamp(*args, **kwargs))