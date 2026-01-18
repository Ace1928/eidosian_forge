from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ReplicationScheduleValueValuesEnum(_messages.Enum):
    """Required. Indicates the schedule for replication.

    Values:
      REPLICATION_SCHEDULE_UNSPECIFIED: Unspecified ReplicationSchedule
      EVERY_10_MINUTES: Replication happens once every 10 minutes.
      HOURLY: Replication happens once every hour.
      DAILY: Replication happens once every day.
    """
    REPLICATION_SCHEDULE_UNSPECIFIED = 0
    EVERY_10_MINUTES = 1
    HOURLY = 2
    DAILY = 3