from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BackupReencryptionConfig(_messages.Message):
    """Backup Reencryption Config

  Enums:
    BackupTypeValueValuesEnum: Type of backups users want to re-encrypt.

  Fields:
    backupLimit: Backup re-encryption limit
    backupType: Type of backups users want to re-encrypt.
  """

    class BackupTypeValueValuesEnum(_messages.Enum):
        """Type of backups users want to re-encrypt.

    Values:
      BACKUP_TYPE_UNSPECIFIED: Unknown backup type, will be defaulted to
        AUTOMATIC backup type
      AUTOMATED: Reencrypt automatic backups
      ON_DEMAND: Reencrypt on-demand backups
    """
        BACKUP_TYPE_UNSPECIFIED = 0
        AUTOMATED = 1
        ON_DEMAND = 2
    backupLimit = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    backupType = _messages.EnumField('BackupTypeValueValuesEnum', 2)