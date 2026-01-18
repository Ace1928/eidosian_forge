from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupApplianceLockInfo(_messages.Message):
    """BackupApplianceLockInfo contains metadata about the backupappliance that
  created the lock.

  Fields:
    backupApplianceId: Required. The ID of the backup/recovery appliance that
      created this lock.
    backupApplianceName: Required. The name of the backup/recovery appliance
      that created this lock.
    backupImage: The image name that depends on this Backup.
    jobName: The job name on the backup/recovery appliance that created this
      lock.
    lockReason: Required. The reason for the lock: e.g.
      MOUNT/RESTORE/BACKUP/etc. The value of this string is only meaningful to
      the client and it is not interpreted by the BackupVault service.
    slaId: The SLA on the backup/recovery appliance that owns the lock.
  """
    backupApplianceId = _messages.IntegerField(1)
    backupApplianceName = _messages.StringField(2)
    backupImage = _messages.StringField(3)
    jobName = _messages.StringField(4)
    lockReason = _messages.StringField(5)
    slaId = _messages.IntegerField(6)