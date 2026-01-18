from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CopyBackupRequest(_messages.Message):
    """The request for CopyBackup.

  Fields:
    backupId: Required. The id of the backup copy. The `backup_id` appended to
      `parent` forms the full backup_uri of the form
      `projects//instances//backups/`.
    encryptionConfig: Optional. The encryption configuration used to encrypt
      the backup. If this field is not specified, the backup will use the same
      encryption configuration as the source backup by default, namely
      encryption_type = `USE_CONFIG_DEFAULT_OR_BACKUP_ENCRYPTION`.
    expireTime: Required. The expiration time of the backup in microsecond
      granularity. The expiration time must be at least 6 hours and at most
      366 days from the `create_time` of the source backup. Once the
      `expire_time` has passed, the backup is eligible to be automatically
      deleted by Cloud Spanner to free the resources used by the backup.
    sourceBackup: Required. The source backup to be copied. The source backup
      needs to be in READY state for it to be copied. Once CopyBackup is in
      progress, the source backup cannot be deleted or cleaned up on
      expiration until CopyBackup is finished. Values are of the form:
      `projects//instances//backups/`.
  """
    backupId = _messages.StringField(1)
    encryptionConfig = _messages.MessageField('CopyBackupEncryptionConfig', 2)
    expireTime = _messages.StringField(3)
    sourceBackup = _messages.StringField(4)