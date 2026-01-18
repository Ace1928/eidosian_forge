from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WindowValueValuesEnum(_messages.Enum):
    """Optional. Window associated with the health report. Default: ONE_HOUR

    Values:
      HEALTH_WINDOW_UNSPECIFIED: Invalid window
      ONE_HOUR: 1 hour
      ONE_DAY: 1 day
      ONE_WEEK: 1 week
    """
    HEALTH_WINDOW_UNSPECIFIED = 0
    ONE_HOUR = 1
    ONE_DAY = 2
    ONE_WEEK = 3