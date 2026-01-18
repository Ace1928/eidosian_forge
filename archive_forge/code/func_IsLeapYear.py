from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
def IsLeapYear(year):
    """Returns True if year is a leap year.

  Cheaper than `import calendar` because its the only thing needed here.

  Args:
    year: The 4 digit year.

  Returns:
    True if year is a leap year.
  """
    return year % 400 == 0 or (year % 100 != 0 and year % 4 == 0)