from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
def _MatchingDays(self, year, month):
    """Returns matching days for the given year and month.

    For the given year and month, return the days that match this instance's
    day specification, based on either (a) the ordinals and weekdays, or
    (b) the explicitly specified monthdays.  If monthdays are specified,
    dates that fall outside the range of the month will not be returned.

    Arguments:
      year: the year as an integer
      month: the month as an integer, in range 1-12

    Returns:
      a list of matching days, as ints in range 1-31
    """
    start_day, last_day = calendar.monthrange(year, month)
    if self.monthdays:
        return sorted([day for day in self.monthdays if day <= last_day])
    out_days = []
    start_day = (start_day + 1) % 7
    for ordinal in self.ordinals:
        for weekday in self.weekdays:
            day = (weekday - start_day) % 7 + 1
            day += 7 * (ordinal - 1)
            if day <= last_day:
                out_days.append(day)
    return sorted(out_days)