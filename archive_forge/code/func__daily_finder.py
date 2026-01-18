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
def _daily_finder(vmin, vmax, freq: BaseOffset) -> np.ndarray:
    dtype_code = freq._period_dtype_code
    periodsperday, periodspermonth, periodsperyear = _get_periods_per_ymd(freq)
    vmin_orig = vmin
    vmin, vmax = (int(vmin), int(vmax))
    span = vmax - vmin + 1
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'Period with BDay freq is deprecated', category=FutureWarning)
        warnings.filterwarnings('ignore', 'PeriodDtype\\[B\\] is deprecated', category=FutureWarning)
        dates_ = period_range(start=Period(ordinal=vmin, freq=freq), end=Period(ordinal=vmax, freq=freq), freq=freq)
    info = np.zeros(span, dtype=[('val', np.int64), ('maj', bool), ('min', bool), ('fmt', '|S20')])
    info['val'][:] = dates_.asi8
    info['fmt'][:] = ''
    info['maj'][[0, -1]] = True
    info_maj = info['maj']
    info_min = info['min']
    info_fmt = info['fmt']

    def first_label(label_flags):
        if label_flags[0] == 0 and label_flags.size > 1 and (vmin_orig % 1 > 0.0):
            return label_flags[1]
        else:
            return label_flags[0]
    if span <= periodspermonth:
        day_start = _period_break(dates_, 'day')
        month_start = _period_break(dates_, 'month')
        year_start = _period_break(dates_, 'year')

        def _hour_finder(label_interval: int, force_year_start: bool) -> None:
            target = dates_.hour
            mask = _period_break_mask(dates_, 'hour')
            info_maj[day_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = '%H:%M'
            info_fmt[day_start] = '%H:%M\n%d-%b'
            info_fmt[year_start] = '%H:%M\n%d-%b\n%Y'
            if force_year_start and (not has_level_label(year_start, vmin_orig)):
                info_fmt[first_label(day_start)] = '%H:%M\n%d-%b\n%Y'

        def _minute_finder(label_interval: int) -> None:
            target = dates_.minute
            hour_start = _period_break(dates_, 'hour')
            mask = _period_break_mask(dates_, 'minute')
            info_maj[hour_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = '%H:%M'
            info_fmt[day_start] = '%H:%M\n%d-%b'
            info_fmt[year_start] = '%H:%M\n%d-%b\n%Y'

        def _second_finder(label_interval: int) -> None:
            target = dates_.second
            minute_start = _period_break(dates_, 'minute')
            mask = _period_break_mask(dates_, 'second')
            info_maj[minute_start] = True
            info_min[mask & (target % label_interval == 0)] = True
            info_fmt[mask & (target % label_interval == 0)] = '%H:%M:%S'
            info_fmt[day_start] = '%H:%M:%S\n%d-%b'
            info_fmt[year_start] = '%H:%M:%S\n%d-%b\n%Y'
        if span < periodsperday / 12000:
            _second_finder(1)
        elif span < periodsperday / 6000:
            _second_finder(2)
        elif span < periodsperday / 2400:
            _second_finder(5)
        elif span < periodsperday / 1200:
            _second_finder(10)
        elif span < periodsperday / 800:
            _second_finder(15)
        elif span < periodsperday / 400:
            _second_finder(30)
        elif span < periodsperday / 150:
            _minute_finder(1)
        elif span < periodsperday / 70:
            _minute_finder(2)
        elif span < periodsperday / 24:
            _minute_finder(5)
        elif span < periodsperday / 12:
            _minute_finder(15)
        elif span < periodsperday / 6:
            _minute_finder(30)
        elif span < periodsperday / 2.5:
            _hour_finder(1, False)
        elif span < periodsperday / 1.5:
            _hour_finder(2, False)
        elif span < periodsperday * 1.25:
            _hour_finder(3, False)
        elif span < periodsperday * 2.5:
            _hour_finder(6, True)
        elif span < periodsperday * 4:
            _hour_finder(12, True)
        else:
            info_maj[month_start] = True
            info_min[day_start] = True
            info_fmt[day_start] = '%d'
            info_fmt[month_start] = '%d\n%b'
            info_fmt[year_start] = '%d\n%b\n%Y'
            if not has_level_label(year_start, vmin_orig):
                if not has_level_label(month_start, vmin_orig):
                    info_fmt[first_label(day_start)] = '%d\n%b\n%Y'
                else:
                    info_fmt[first_label(month_start)] = '%d\n%b\n%Y'
    elif span <= periodsperyear // 4:
        month_start = _period_break(dates_, 'month')
        info_maj[month_start] = True
        if dtype_code < FreqGroup.FR_HR.value:
            info['min'] = True
        else:
            day_start = _period_break(dates_, 'day')
            info['min'][day_start] = True
        week_start = _period_break(dates_, 'week')
        year_start = _period_break(dates_, 'year')
        info_fmt[week_start] = '%d'
        info_fmt[month_start] = '\n\n%b'
        info_fmt[year_start] = '\n\n%b\n%Y'
        if not has_level_label(year_start, vmin_orig):
            if not has_level_label(month_start, vmin_orig):
                info_fmt[first_label(week_start)] = '\n\n%b\n%Y'
            else:
                info_fmt[first_label(month_start)] = '\n\n%b\n%Y'
    elif span <= 1.15 * periodsperyear:
        year_start = _period_break(dates_, 'year')
        month_start = _period_break(dates_, 'month')
        week_start = _period_break(dates_, 'week')
        info_maj[month_start] = True
        info_min[week_start] = True
        info_min[year_start] = False
        info_min[month_start] = False
        info_fmt[month_start] = '%b'
        info_fmt[year_start] = '%b\n%Y'
        if not has_level_label(year_start, vmin_orig):
            info_fmt[first_label(month_start)] = '%b\n%Y'
    elif span <= 2.5 * periodsperyear:
        year_start = _period_break(dates_, 'year')
        quarter_start = _period_break(dates_, 'quarter')
        month_start = _period_break(dates_, 'month')
        info_maj[quarter_start] = True
        info_min[month_start] = True
        info_fmt[quarter_start] = '%b'
        info_fmt[year_start] = '%b\n%Y'
    elif span <= 4 * periodsperyear:
        year_start = _period_break(dates_, 'year')
        month_start = _period_break(dates_, 'month')
        info_maj[year_start] = True
        info_min[month_start] = True
        info_min[year_start] = False
        month_break = dates_[month_start].month
        jan_or_jul = month_start[(month_break == 1) | (month_break == 7)]
        info_fmt[jan_or_jul] = '%b'
        info_fmt[year_start] = '%b\n%Y'
    elif span <= 11 * periodsperyear:
        year_start = _period_break(dates_, 'year')
        quarter_start = _period_break(dates_, 'quarter')
        info_maj[year_start] = True
        info_min[quarter_start] = True
        info_min[year_start] = False
        info_fmt[year_start] = '%Y'
    else:
        year_start = _period_break(dates_, 'year')
        year_break = dates_[year_start].year
        nyears = span / periodsperyear
        min_anndef, maj_anndef = _get_default_annual_spacing(nyears)
        major_idx = year_start[year_break % maj_anndef == 0]
        info_maj[major_idx] = True
        minor_idx = year_start[year_break % min_anndef == 0]
        info_min[minor_idx] = True
        info_fmt[major_idx] = '%Y'
    return info