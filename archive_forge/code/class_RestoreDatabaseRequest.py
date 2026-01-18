from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RestoreDatabaseRequest(_messages.Message):
    """The request for RestoreDatabase.

  Fields:
    backup: Name of the backup from which to restore. Values are of the form
      `projects//instances//backups/`.
    databaseId: Required. The id of the database to create and restore to.
      This database must not already exist. The `database_id` appended to
      `parent` forms the full database name of the form
      `projects//instances//databases/`.
    encryptionConfig: Optional. An encryption configuration describing the
      encryption type and key resources in Cloud KMS used to encrypt/decrypt
      the database to restore to. If this field is not specified, the restored
      database will use the same encryption configuration as the backup by
      default, namely encryption_type =
      `USE_CONFIG_DEFAULT_OR_BACKUP_ENCRYPTION`.
  """
    backup = _messages.StringField(1)
    databaseId = _messages.StringField(2)
    encryptionConfig = _messages.MessageField('RestoreDatabaseEncryptionConfig', 3)