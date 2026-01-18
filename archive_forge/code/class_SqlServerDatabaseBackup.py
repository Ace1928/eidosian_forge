from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerDatabaseBackup(_messages.Message):
    """Specifies the backup details for a single database in Cloud Storage for
  homogeneous migration to Cloud SQL for SQL Server.

  Fields:
    database: Required. Name of a SQL Server database for which to define
      backup configuration.
    encryptionOptions: Optional. Encryption settings for the database.
      Required if provided database backups are encrypted. Encryption settings
      include path to certificate, path to certificate private key, and key
      password.
    encryptionOptionsOverride: Optional. Encryption settings for the database.
      Required if provided database backups are encrypted. Encryption settings
      include path to certificate, path to certificate private key, and key
      password. To be deprecated.
  """
    database = _messages.StringField(1)
    encryptionOptions = _messages.MessageField('SqlServerEncryptionOptions', 2)
    encryptionOptionsOverride = _messages.MessageField('SqlServerEncryptionOptions', 3)