from __future__ import absolute_import
from six.moves import input
from decimal import Decimal
import re
from gslib.exception import CommandException
from gslib.lazy_wrapper import LazyWrapper
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def RetentionInSeconds(pattern):
    """Converts a retention period string pattern to equivalent seconds.

  Args:
    pattern: a string pattern that represents a retention period.

  Returns:
    Returns the retention period in seconds. If the pattern does not match
  """
    seconds = None
    year_match = RetentionInYearsMatch(pattern)
    month_match = RetentionInMonthsMatch(pattern)
    day_match = RetentionInDaysMatch(pattern)
    second_match = RetentionInSecondsMatch(pattern)
    if year_match:
        seconds = YearsToSeconds(int(year_match.group('number')))
    elif month_match:
        seconds = MonthsToSeconds(int(month_match.group('number')))
    elif day_match:
        seconds = DaysToSeconds(int(day_match.group('number')))
    elif second_match:
        seconds = int(second_match.group('number'))
    else:
        raise CommandException('Incorrect retention period specified. Please use one of the following formats to specify the retention period : <number>y, <number>m, <number>d, <number>s.')
    return seconds