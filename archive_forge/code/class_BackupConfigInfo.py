from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BackupConfigInfo(_messages.Message):
    """BackupConfigInfo has information about how the resource is configured
  for Backup and about the most recent backup to this vault.

  Enums:
    LastBackupStateValueValuesEnum: Output only. The status of the last backup
      to this BackupVault

  Fields:
    backupApplianceBackupConfig: Configuration for an application backed up by
      a Backup Appliance.
    gcpBackupConfig: Configuration for a GCP resource.
    lastBackupError: Output only. If the last backup failed, this field has
      the error message.
    lastBackupState: Output only. The status of the last backup to this
      BackupVault
    lastSuccessfulBackupConsistencyTime: Output only. If the last backup were
      successful, this field has the consistency date.
  """

    class LastBackupStateValueValuesEnum(_messages.Enum):
        """Output only. The status of the last backup to this BackupVault

    Values:
      LAST_BACKUP_STATE_UNSPECIFIED: Status not set.
      FIRST_BACKUP_PENDING: The first backup has not yet completed
      SUCCEEDED: The most recent backup was successful
      FAILED: The most recent backup failed
      PERMISSION_DENIED: The most recent backup could not be run/failed
        because of the lack of permissions
    """
        LAST_BACKUP_STATE_UNSPECIFIED = 0
        FIRST_BACKUP_PENDING = 1
        SUCCEEDED = 2
        FAILED = 3
        PERMISSION_DENIED = 4
    backupApplianceBackupConfig = _messages.MessageField('BackupApplianceBackupConfig', 1)
    gcpBackupConfig = _messages.MessageField('GcpBackupConfig', 2)
    lastBackupError = _messages.MessageField('Status', 3)
    lastBackupState = _messages.EnumField('LastBackupStateValueValuesEnum', 4)
    lastSuccessfulBackupConsistencyTime = _messages.StringField(5)