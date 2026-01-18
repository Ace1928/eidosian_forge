from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupLock(_messages.Message):
    """BackupLock represents a single lock on a Backup resource. An unexpired
  lock on a Backup prevents the Backup from being deleted.

  Fields:
    backupApplianceLockInfo: If the client is a backup and recovery appliance,
      this contains metadata about why the lock exists.
    lockUntilTime: Required. The time after which this lock is not considered
      valid and will no longer protect the Backup from deletion.
    serviceLockInfo: Output only. Contains metadata about the lock exist for
      GCP native backups.
  """
    backupApplianceLockInfo = _messages.MessageField('BackupApplianceLockInfo', 1)
    lockUntilTime = _messages.StringField(2)
    serviceLockInfo = _messages.MessageField('ServiceLockInfo', 3)