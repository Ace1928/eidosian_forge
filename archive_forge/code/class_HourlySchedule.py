from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HourlySchedule(_messages.Message):
    """Make a snapshot every hour e.g. at 04:00, 05:00, 06:00.

  Fields:
    minute: Set the minute of the hour to start the snapshot (0-59), defaults
      to the top of the hour (0).
    snapshotsToKeep: The maximum number of Snapshots to keep for the hourly
      schedule
  """
    minute = _messages.FloatField(1)
    snapshotsToKeep = _messages.FloatField(2)