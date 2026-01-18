from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RdbSnapshotPeriodValueValuesEnum(_messages.Enum):
    """Optional. Period between RDB snapshots.

    Values:
      SNAPSHOT_PERIOD_UNSPECIFIED: Not set.
      ONE_HOUR: One hour.
      SIX_HOURS: Six hours.
      TWELVE_HOURS: Twelve hours.
      TWENTY_FOUR_HOURS: Twenty four hours.
    """
    SNAPSHOT_PERIOD_UNSPECIFIED = 0
    ONE_HOUR = 1
    SIX_HOURS = 2
    TWELVE_HOURS = 3
    TWENTY_FOUR_HOURS = 4