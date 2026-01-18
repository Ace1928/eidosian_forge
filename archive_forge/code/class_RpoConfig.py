from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RpoConfig(_messages.Message):
    """Defines RPO scheduling configuration for automatically creating Backups
  via this BackupPlan.

  Fields:
    exclusionWindows: Optional. User specified time windows during which
      backup can NOT happen for this BackupPlan - backups should start and
      finish outside of any given exclusion window. Note: backup jobs will be
      scheduled to start and finish outside the duration of the window as much
      as possible, but running jobs will not get canceled when it runs into
      the window. All the time and date values in exclusion_windows entry in
      the API are in UTC. We only allow <=1 recurrence (daily or weekly)
      exclusion window for a BackupPlan while no restriction on number of
      single occurrence windows.
    targetRpoMinutes: Required. Defines the target RPO for the BackupPlan in
      minutes, which means the target maximum data loss in time that is
      acceptable for this BackupPlan. This must be at least 60, i.e., 1 hour,
      and at most 86400, i.e., 60 days.
  """
    exclusionWindows = _messages.MessageField('ExclusionWindow', 1, repeated=True)
    targetRpoMinutes = _messages.IntegerField(2, variant=_messages.Variant.INT32)