from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class BakImportOptionsValue(_messages.Message):
    """Import parameters specific to SQL Server .BAK files

    Enums:
      BakTypeValueValuesEnum: Type of the bak content, FULL or DIFF.

    Messages:
      EncryptionOptionsValue: A EncryptionOptionsValue object.

    Fields:
      bakType: Type of the bak content, FULL or DIFF.
      encryptionOptions: A EncryptionOptionsValue attribute.
      noRecovery: Whether or not the backup importing will restore database
        with NORECOVERY option Applies only to Cloud SQL for SQL Server.
      recoveryOnly: Whether or not the backup importing request will just
        bring database online without downloading Bak content only one of
        "no_recovery" and "recovery_only" can be true otherwise error will
        return. Applies only to Cloud SQL for SQL Server.
      stopAt: Optional. The timestamp when the import should stop. This
        timestamp is in the [RFC 3339](https://tools.ietf.org/html/rfc3339)
        format (for example, `2023-10-01T16:19:00.094`). This field is
        equivalent to the STOPAT keyword and applies to Cloud SQL for SQL
        Server only.
      stopAtMark: Optional. The marked transaction where the import should
        stop. This field is equivalent to the STOPATMARK keyword and applies
        to Cloud SQL for SQL Server only.
      striped: Whether or not the backup set being restored is striped.
        Applies only to Cloud SQL for SQL Server.
    """

    class BakTypeValueValuesEnum(_messages.Enum):
        """Type of the bak content, FULL or DIFF.

      Values:
        BAK_TYPE_UNSPECIFIED: Default type.
        FULL: Full backup.
        DIFF: Differential backup.
        TLOG: SQL Server Transaction Log
      """
        BAK_TYPE_UNSPECIFIED = 0
        FULL = 1
        DIFF = 2
        TLOG = 3

    class EncryptionOptionsValue(_messages.Message):
        """A EncryptionOptionsValue object.

      Fields:
        certPath: Path to the Certificate (.cer) in Cloud Storage, in the form
          `gs://bucketName/fileName`. The instance must have write permissions
          to the bucket and read access to the file.
        pvkPassword: Password that encrypts the private key
        pvkPath: Path to the Certificate Private Key (.pvk) in Cloud Storage,
          in the form `gs://bucketName/fileName`. The instance must have write
          permissions to the bucket and read access to the file.
      """
        certPath = _messages.StringField(1)
        pvkPassword = _messages.StringField(2)
        pvkPath = _messages.StringField(3)
    bakType = _messages.EnumField('BakTypeValueValuesEnum', 1)
    encryptionOptions = _messages.MessageField('EncryptionOptionsValue', 2)
    noRecovery = _messages.BooleanField(3)
    recoveryOnly = _messages.BooleanField(4)
    stopAt = _messages.StringField(5)
    stopAtMark = _messages.StringField(6)
    striped = _messages.BooleanField(7)