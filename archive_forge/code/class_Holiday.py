from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
class Holiday:
    """
    Class that defines a holiday with start/end dates and rules
    for observance.
    """
    start_date: Timestamp | None
    end_date: Timestamp | None
    days_of_week: tuple[int, ...] | None

    def __init__(self, name: str, year=None, month=None, day=None, offset=None, observance=None, start_date=None, end_date=None, days_of_week=None) -> None:
        """
        Parameters
        ----------
        name : str
            Name of the holiday , defaults to class name
        offset : array of pandas.tseries.offsets or
                class from pandas.tseries.offsets
            computes offset from date
        observance: function
            computes when holiday is given a pandas Timestamp
        days_of_week:
            provide a tuple of days e.g  (0,1,2,3,) for Monday Through Thursday
            Monday=0,..,Sunday=6

        Examples
        --------
        >>> from dateutil.relativedelta import MO

        >>> USMemorialDay = pd.tseries.holiday.Holiday(
        ...     "Memorial Day", month=5, day=31, offset=pd.DateOffset(weekday=MO(-1))
        ... )
        >>> USMemorialDay
        Holiday: Memorial Day (month=5, day=31, offset=<DateOffset: weekday=MO(-1)>)

        >>> USLaborDay = pd.tseries.holiday.Holiday(
        ...     "Labor Day", month=9, day=1, offset=pd.DateOffset(weekday=MO(1))
        ... )
        >>> USLaborDay
        Holiday: Labor Day (month=9, day=1, offset=<DateOffset: weekday=MO(+1)>)

        >>> July3rd = pd.tseries.holiday.Holiday("July 3rd", month=7, day=3)
        >>> July3rd
        Holiday: July 3rd (month=7, day=3, )

        >>> NewYears = pd.tseries.holiday.Holiday(
        ...     "New Years Day", month=1,  day=1,
        ...      observance=pd.tseries.holiday.nearest_workday
        ... )
        >>> NewYears  # doctest: +SKIP
        Holiday: New Years Day (
            month=1, day=1, observance=<function nearest_workday at 0x66545e9bc440>
        )

        >>> July3rd = pd.tseries.holiday.Holiday(
        ...     "July 3rd", month=7, day=3,
        ...     days_of_week=(0, 1, 2, 3)
        ... )
        >>> July3rd
        Holiday: July 3rd (month=7, day=3, )
        """
        if offset is not None and observance is not None:
            raise NotImplementedError('Cannot use both offset and observance.')
        self.name = name
        self.year = year
        self.month = month
        self.day = day
        self.offset = offset
        self.start_date = Timestamp(start_date) if start_date is not None else start_date
        self.end_date = Timestamp(end_date) if end_date is not None else end_date
        self.observance = observance
        assert days_of_week is None or type(days_of_week) == tuple
        self.days_of_week = days_of_week

    def __repr__(self) -> str:
        info = ''
        if self.year is not None:
            info += f'year={self.year}, '
        info += f'month={self.month}, day={self.day}, '
        if self.offset is not None:
            info += f'offset={self.offset}'
        if self.observance is not None:
            info += f'observance={self.observance}'
        repr = f'Holiday: {self.name} ({info})'
        return repr

    def dates(self, start_date, end_date, return_name: bool=False) -> Series | DatetimeIndex:
        """
        Calculate holidays observed between start date and end date

        Parameters
        ----------
        start_date : starting date, datetime-like, optional
        end_date : ending date, datetime-like, optional
        return_name : bool, optional, default=False
            If True, return a series that has dates and holiday names.
            False will only return dates.

        Returns
        -------
        Series or DatetimeIndex
            Series if return_name is True
        """
        start_date = Timestamp(start_date)
        end_date = Timestamp(end_date)
        filter_start_date = start_date
        filter_end_date = end_date
        if self.year is not None:
            dt = Timestamp(datetime(self.year, self.month, self.day))
            dti = DatetimeIndex([dt])
            if return_name:
                return Series(self.name, index=dti)
            else:
                return dti
        dates = self._reference_dates(start_date, end_date)
        holiday_dates = self._apply_rule(dates)
        if self.days_of_week is not None:
            holiday_dates = holiday_dates[np.isin(holiday_dates.dayofweek, self.days_of_week).ravel()]
        if self.start_date is not None:
            filter_start_date = max(self.start_date.tz_localize(filter_start_date.tz), filter_start_date)
        if self.end_date is not None:
            filter_end_date = min(self.end_date.tz_localize(filter_end_date.tz), filter_end_date)
        holiday_dates = holiday_dates[(holiday_dates >= filter_start_date) & (holiday_dates <= filter_end_date)]
        if return_name:
            return Series(self.name, index=holiday_dates)
        return holiday_dates

    def _reference_dates(self, start_date: Timestamp, end_date: Timestamp) -> DatetimeIndex:
        """
        Get reference dates for the holiday.

        Return reference dates for the holiday also returning the year
        prior to the start_date and year following the end_date.  This ensures
        that any offsets to be applied will yield the holidays within
        the passed in dates.
        """
        if self.start_date is not None:
            start_date = self.start_date.tz_localize(start_date.tz)
        if self.end_date is not None:
            end_date = self.end_date.tz_localize(start_date.tz)
        year_offset = DateOffset(years=1)
        reference_start_date = Timestamp(datetime(start_date.year - 1, self.month, self.day))
        reference_end_date = Timestamp(datetime(end_date.year + 1, self.month, self.day))
        dates = date_range(start=reference_start_date, end=reference_end_date, freq=year_offset, tz=start_date.tz)
        return dates

    def _apply_rule(self, dates: DatetimeIndex) -> DatetimeIndex:
        """
        Apply the given offset/observance to a DatetimeIndex of dates.

        Parameters
        ----------
        dates : DatetimeIndex
            Dates to apply the given offset/observance rule

        Returns
        -------
        Dates with rules applied
        """
        if dates.empty:
            return dates.copy()
        if self.observance is not None:
            return dates.map(lambda d: self.observance(d))
        if self.offset is not None:
            if not isinstance(self.offset, list):
                offsets = [self.offset]
            else:
                offsets = self.offset
            for offset in offsets:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', PerformanceWarning)
                    dates += offset
        return dates