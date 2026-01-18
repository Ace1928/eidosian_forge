from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerHomogeneousMigrationJobConfig(_messages.Message):
    """Configuration for homogeneous migration to Cloud SQL for SQL Server.

  Messages:
    DatabaseDetailsValue: Optional. Backup details per database in Cloud
      Storage.

  Fields:
    backupFilePattern: Required. Pattern that describes the default backup
      naming strategy. The specified pattern should ensure lexicographical
      order of backups. The pattern must define one of the following capture
      group sets: Capture group set #1 yy/yyyy - year, 2 or 4 digits mm -
      month number, 1-12 dd - day of month, 1-31 hh - hour of day, 00-23 mi -
      minutes, 00-59 ss - seconds, 00-59 Example: For backup file
      TestDB_20230802_155400.trn, use pattern:
      (?.*)_backup_(?\\d{4})(?\\d{2})(?\\d{2})_(?\\d{2})(?\\d{2})(?\\d{2}).trn
      Capture group set #2 timestamp - unix timestamp Example: For backup file
      TestDB.1691448254.trn, use pattern: (?.*)\\.(?\\d*).trn or
      (?.*)\\.(?\\d*).trn
    databaseBackups: Required. Backup details per database in Cloud Storage.
    databaseDetails: Optional. Backup details per database in Cloud Storage.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DatabaseDetailsValue(_messages.Message):
        """Optional. Backup details per database in Cloud Storage.

    Messages:
      AdditionalProperty: An additional property for a DatabaseDetailsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DatabaseDetailsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DatabaseDetailsValue object.

      Fields:
        key: Name of the additional property.
        value: A SqlServerDatabaseDetails attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('SqlServerDatabaseDetails', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    backupFilePattern = _messages.StringField(1)
    databaseBackups = _messages.MessageField('SqlServerDatabaseBackup', 2, repeated=True)
    databaseDetails = _messages.MessageField('DatabaseDetailsValue', 3)