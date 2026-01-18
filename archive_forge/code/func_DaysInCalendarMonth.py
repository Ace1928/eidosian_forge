from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
def DaysInCalendarMonth(year, month):
    """Returns the number of days in the given month and calendar year.

  Args:
    year: The 4 digit calendar year.
    month: The month number 1..12.

  Returns:
    The number of days in the given month and calendar year.
  """
    return _DAYS_IN_MONTH[month - 1] + (1 if month == 2 and IsLeapYear(year) else 0)