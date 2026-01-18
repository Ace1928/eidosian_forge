from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class TimeRangePeriodValueValuesEnum(_messages.Enum):
    """Restricts the query to the specified time range.

    Values:
      PERIOD_UNSPECIFIED: Do not use.
      PERIOD_1_HOUR: Retrieve data for the last hour. Recommended minimum
        timed count duration: 1 min.
      PERIOD_6_HOURS: Retrieve data for the last 6 hours. Recommended minimum
        timed count duration: 10 min.
      PERIOD_1_DAY: Retrieve data for the last day. Recommended minimum timed
        count duration: 1 hour.
      PERIOD_1_WEEK: Retrieve data for the last week. Recommended minimum
        timed count duration: 6 hours.
      PERIOD_30_DAYS: Retrieve data for the last 30 days. Recommended minimum
        timed count duration: 1 day.
    """
    PERIOD_UNSPECIFIED = 0
    PERIOD_1_HOUR = 1
    PERIOD_6_HOURS = 2
    PERIOD_1_DAY = 3
    PERIOD_1_WEEK = 4
    PERIOD_30_DAYS = 5