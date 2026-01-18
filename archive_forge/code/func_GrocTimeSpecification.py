from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import calendar
import datetime
from . import groc
def GrocTimeSpecification(schedule, timezone=None):
    """Factory function.

  Turns a schedule specification into a TimeSpecification.

  Arguments:
    schedule: the schedule specification, as a string
    timezone: the optional timezone as a string for this specification. Defaults
      to 'UTC' - valid entries are things like 'Australia/Victoria' or
      'PST8PDT'.

  Returns:
    a TimeSpecification instance
  """
    parser = groc.CreateParser(schedule)
    parser.timespec()
    if parser.period_string:
        return IntervalTimeSpecification(parser.interval_mins, parser.period_string, parser.synchronized, parser.start_time_string, parser.end_time_string, timezone)
    else:
        return SpecificTimeSpecification(parser.ordinal_set, parser.weekday_set, parser.month_set, parser.monthday_set, parser.time_string, timezone)