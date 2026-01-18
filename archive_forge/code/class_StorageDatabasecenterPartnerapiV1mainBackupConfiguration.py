from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StorageDatabasecenterPartnerapiV1mainBackupConfiguration(_messages.Message):
    """Configuration for automatic backups

  Fields:
    automatedBackupEnabled: Whether customer visible automated backups are
      enabled on the instance.
    backupRetentionSettings: Backup retention settings.
    pointInTimeRecoveryEnabled: Whether point-in-time recovery is enabled.
      This is optional field, if the database service does not have this
      feature or metadata is not available in control plane, this can be
      omitted.
  """
    automatedBackupEnabled = _messages.BooleanField(1)
    backupRetentionSettings = _messages.MessageField('StorageDatabasecenterPartnerapiV1mainRetentionSettings', 2)
    pointInTimeRecoveryEnabled = _messages.BooleanField(3)