from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class FrequencyValueValuesEnum(_messages.Enum):
    """Required. The frequency unit of this recurring schedule.

    Values:
      FREQUENCY_UNSPECIFIED: Invalid. A frequency must be specified.
      WEEKLY: Indicates that the frequency of recurrence should be expressed
        in terms of weeks.
      MONTHLY: Indicates that the frequency of recurrence should be expressed
        in terms of months.
      DAILY: Indicates that the frequency of recurrence should be expressed in
        terms of days.
    """
    FREQUENCY_UNSPECIFIED = 0
    WEEKLY = 1
    MONTHLY = 2
    DAILY = 3