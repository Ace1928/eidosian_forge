from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScheduledSnapshotsConfig(_messages.Message):
    """The configuration for scheduled snapshot creation mechanism.

  Fields:
    enabled: Optional. Whether scheduled snapshots creation is enabled.
    snapshotCreationSchedule: Optional. The cron expression representing the
      time when snapshots creation mechanism runs. This field is subject to
      additional validation around frequency of execution.
    snapshotLocation: Optional. The Cloud Storage location for storing
      automatically created snapshots.
    timeZone: Optional. Time zone that sets the context to interpret
      snapshot_creation_schedule.
  """
    enabled = _messages.BooleanField(1)
    snapshotCreationSchedule = _messages.StringField(2)
    snapshotLocation = _messages.StringField(3)
    timeZone = _messages.StringField(4)