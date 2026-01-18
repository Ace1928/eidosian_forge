from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WindowSizeValueValuesEnum(_messages.Enum):
    """Time buckets to group the stats by.

    Values:
      WINDOW_SIZE_UNSPECIFIED: Unspecified window size. Default is 1 hour.
      MINUTE: 1 Minute window
      HOUR: 1 Hour window
      DAY: 1 Day window
      MONTH: 1 Month window
    """
    WINDOW_SIZE_UNSPECIFIED = 0
    MINUTE = 1
    HOUR = 2
    DAY = 3
    MONTH = 4