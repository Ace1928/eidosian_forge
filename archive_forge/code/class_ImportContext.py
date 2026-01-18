from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ImportContext(_messages.Message):
    """Database instance import context.

  Enums:
    FileTypeValueValuesEnum: The file type for the specified uri. * `SQL`: The
      file contains SQL statements. * `CSV`: The file contains CSV data. *
      `BAK`: The file contains backup data for a SQL Server instance.

  Messages:
    BakImportOptionsValue: Import parameters specific to SQL Server .BAK files
    CsvImportOptionsValue: Options for importing data as CSV.
    SqlImportOptionsValue: Optional. Options for importing data from SQL
      statements.

  Fields:
    bakImportOptions: Import parameters specific to SQL Server .BAK files
    csvImportOptions: Options for importing data as CSV.
    database: The target database for the import. If `fileType` is `SQL`, this
      field is required only if the import file does not specify a database,
      and is overridden by any database specification in the import file. If
      `fileType` is `CSV`, one database must be specified.
    fileType: The file type for the specified uri. * `SQL`: The file contains
      SQL statements. * `CSV`: The file contains CSV data. * `BAK`: The file
      contains backup data for a SQL Server instance.
    importUser: The PostgreSQL user for this import operation. PostgreSQL
      instances only.
    kind: This is always `sql#importContext`.
    sqlImportOptions: Optional. Options for importing data from SQL
      statements.
    uri: Path to the import file in Cloud Storage, in the form
      `gs://bucketName/fileName`. Compressed gzip files (.gz) are supported
      when `fileType` is `SQL`. The instance must have write permissions to
      the bucket and read access to the file.
  """

    class FileTypeValueValuesEnum(_messages.Enum):
        """The file type for the specified uri. * `SQL`: The file contains SQL
    statements. * `CSV`: The file contains CSV data. * `BAK`: The file
    contains backup data for a SQL Server instance.

    Values:
      SQL_FILE_TYPE_UNSPECIFIED: Unknown file type.
      SQL: File containing SQL statements.
      CSV: File in CSV format.
      BAK: <no description>
    """
        SQL_FILE_TYPE_UNSPECIFIED = 0
        SQL = 1
        CSV = 2
        BAK = 3

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

    class CsvImportOptionsValue(_messages.Message):
        """Options for importing data as CSV.

    Fields:
      columns: The columns to which CSV data is imported. If not specified,
        all columns of the database table are loaded with CSV data.
      escapeCharacter: Specifies the character that should appear before a
        data character that needs to be escaped.
      fieldsTerminatedBy: Specifies the character that separates columns
        within each row (line) of the file.
      linesTerminatedBy: This is used to separate lines. If a line does not
        contain all fields, the rest of the columns are set to their default
        values.
      quoteCharacter: Specifies the quoting character to be used when a data
        value is quoted.
      table: The table to which CSV data is imported.
    """
        columns = _messages.StringField(1, repeated=True)
        escapeCharacter = _messages.StringField(2)
        fieldsTerminatedBy = _messages.StringField(3)
        linesTerminatedBy = _messages.StringField(4)
        quoteCharacter = _messages.StringField(5)
        table = _messages.StringField(6)

    class SqlImportOptionsValue(_messages.Message):
        """Optional. Options for importing data from SQL statements.

    Fields:
      parallel: Optional. Whether or not the import should be parallel.
      threads: Optional. The number of threads to use for parallel import.
    """
        parallel = _messages.BooleanField(1)
        threads = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    bakImportOptions = _messages.MessageField('BakImportOptionsValue', 1)
    csvImportOptions = _messages.MessageField('CsvImportOptionsValue', 2)
    database = _messages.StringField(3)
    fileType = _messages.EnumField('FileTypeValueValuesEnum', 4)
    importUser = _messages.StringField(5)
    kind = _messages.StringField(6)
    sqlImportOptions = _messages.MessageField('SqlImportOptionsValue', 7)
    uri = _messages.StringField(8)