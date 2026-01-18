from __future__ import annotations
from datetime import (
import warnings
from dateutil.relativedelta import (
import numpy as np
from pandas.errors import PerformanceWarning
from pandas import (
from pandas.tseries.offsets import (
class AbstractHolidayCalendar(metaclass=HolidayCalendarMetaClass):
    """
    Abstract interface to create holidays following certain rules.
    """
    rules: list[Holiday] = []
    start_date = Timestamp(datetime(1970, 1, 1))
    end_date = Timestamp(datetime(2200, 12, 31))
    _cache = None

    def __init__(self, name: str='', rules=None) -> None:
        """
        Initializes holiday object with a given set a rules.  Normally
        classes just have the rules defined within them.

        Parameters
        ----------
        name : str
            Name of the holiday calendar, defaults to class name
        rules : array of Holiday objects
            A set of rules used to create the holidays.
        """
        super().__init__()
        if not name:
            name = type(self).__name__
        self.name = name
        if rules is not None:
            self.rules = rules

    def rule_from_name(self, name: str):
        for rule in self.rules:
            if rule.name == name:
                return rule
        return None

    def holidays(self, start=None, end=None, return_name: bool=False):
        """
        Returns a curve with holidays between start_date and end_date

        Parameters
        ----------
        start : starting date, datetime-like, optional
        end : ending date, datetime-like, optional
        return_name : bool, optional
            If True, return a series that has dates and holiday names.
            False will only return a DatetimeIndex of dates.

        Returns
        -------
            DatetimeIndex of holidays
        """
        if self.rules is None:
            raise Exception(f'Holiday Calendar {self.name} does not have any rules specified')
        if start is None:
            start = AbstractHolidayCalendar.start_date
        if end is None:
            end = AbstractHolidayCalendar.end_date
        start = Timestamp(start)
        end = Timestamp(end)
        if self._cache is None or start < self._cache[0] or end > self._cache[1]:
            pre_holidays = [rule.dates(start, end, return_name=True) for rule in self.rules]
            if pre_holidays:
                holidays = concat(pre_holidays)
            else:
                holidays = Series(index=DatetimeIndex([]), dtype=object)
            self._cache = (start, end, holidays.sort_index())
        holidays = self._cache[2]
        holidays = holidays[start:end]
        if return_name:
            return holidays
        else:
            return holidays.index

    @staticmethod
    def merge_class(base, other):
        """
        Merge holiday calendars together. The base calendar
        will take precedence to other. The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        base : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        other : AbstractHolidayCalendar
          instance/subclass or array of Holiday objects
        """
        try:
            other = other.rules
        except AttributeError:
            pass
        if not isinstance(other, list):
            other = [other]
        other_holidays = {holiday.name: holiday for holiday in other}
        try:
            base = base.rules
        except AttributeError:
            pass
        if not isinstance(base, list):
            base = [base]
        base_holidays = {holiday.name: holiday for holiday in base}
        other_holidays.update(base_holidays)
        return list(other_holidays.values())

    def merge(self, other, inplace: bool=False):
        """
        Merge holiday calendars together.  The caller's class
        rules take precedence.  The merge will be done
        based on each holiday's name.

        Parameters
        ----------
        other : holiday calendar
        inplace : bool (default=False)
            If True set rule_table to holidays, else return array of Holidays
        """
        holidays = self.merge_class(self, other)
        if inplace:
            self.rules = holidays
        else:
            return holidays