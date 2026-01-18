from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ExportContext(_messages.Message):
    """Database instance export context.

  Enums:
    FileTypeValueValuesEnum: The file type for the specified uri.

  Messages:
    BakExportOptionsValue: Options for exporting BAK files (SQL Server-only)
    CsvExportOptionsValue: Options for exporting data as CSV. `MySQL` and
      `PostgreSQL` instances only.
    SqlExportOptionsValue: Options for exporting data as SQL statements.

  Fields:
    bakExportOptions: Options for exporting BAK files (SQL Server-only)
    csvExportOptions: Options for exporting data as CSV. `MySQL` and
      `PostgreSQL` instances only.
    databases: Databases to be exported. `MySQL instances:` If `fileType` is
      `SQL` and no database is specified, all databases are exported, except
      for the `mysql` system database. If `fileType` is `CSV`, you can specify
      one database, either by using this property or by using the
      `csvExportOptions.selectQuery` property, which takes precedence over
      this property. `PostgreSQL instances:` You must specify one database to
      be exported. If `fileType` is `CSV`, this database must match the one
      specified in the `csvExportOptions.selectQuery` property. `SQL Server
      instances:` You must specify one database to be exported, and the
      `fileType` must be `BAK`.
    fileType: The file type for the specified uri.
    kind: This is always `sql#exportContext`.
    offload: Option for export offload.
    sqlExportOptions: Options for exporting data as SQL statements.
    uri: The path to the file in Google Cloud Storage where the export will be
      stored. The URI is in the form `gs://bucketName/fileName`. If the file
      already exists, the request succeeds, but the operation fails. If
      `fileType` is `SQL` and the filename ends with .gz, the contents are
      compressed.
  """

    class FileTypeValueValuesEnum(_messages.Enum):
        """The file type for the specified uri.

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

    class BakExportOptionsValue(_messages.Message):
        """Options for exporting BAK files (SQL Server-only)

    Enums:
      BakTypeValueValuesEnum: Type of this bak file will be export, FULL or
        DIFF, SQL Server only

    Fields:
      bakType: Type of this bak file will be export, FULL or DIFF, SQL Server
        only
      copyOnly: Deprecated: copy_only is deprecated. Use differential_base
        instead
      differentialBase: Whether or not the backup can be used as a
        differential base copy_only backup can not be served as differential
        base
      stripeCount: Option for specifying how many stripes to use for the
        export. If blank, and the value of the striped field is true, the
        number of stripes is automatically chosen.
      striped: Whether or not the export should be striped.
    """

        class BakTypeValueValuesEnum(_messages.Enum):
            """Type of this bak file will be export, FULL or DIFF, SQL Server only

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
        bakType = _messages.EnumField('BakTypeValueValuesEnum', 1)
        copyOnly = _messages.BooleanField(2)
        differentialBase = _messages.BooleanField(3)
        stripeCount = _messages.IntegerField(4, variant=_messages.Variant.INT32)
        striped = _messages.BooleanField(5)

    class CsvExportOptionsValue(_messages.Message):
        """Options for exporting data as CSV. `MySQL` and `PostgreSQL` instances
    only.

    Fields:
      escapeCharacter: Specifies the character that should appear before a
        data character that needs to be escaped.
      fieldsTerminatedBy: Specifies the character that separates columns
        within each row (line) of the file.
      linesTerminatedBy: This is used to separate lines. If a line does not
        contain all fields, the rest of the columns are set to their default
        values.
      quoteCharacter: Specifies the quoting character to be used when a data
        value is quoted.
      selectQuery: The select query used to extract the data.
    """
        escapeCharacter = _messages.StringField(1)
        fieldsTerminatedBy = _messages.StringField(2)
        linesTerminatedBy = _messages.StringField(3)
        quoteCharacter = _messages.StringField(4)
        selectQuery = _messages.StringField(5)

    class SqlExportOptionsValue(_messages.Message):
        """Options for exporting data as SQL statements.

    Messages:
      MysqlExportOptionsValue: Options for exporting from MySQL.

    Fields:
      mysqlExportOptions: Options for exporting from MySQL.
      parallel: Optional. Whether or not the export should be parallel.
      schemaOnly: Export only schemas.
      tables: Tables to export, or that were exported, from the specified
        database. If you specify tables, specify one and only one database.
        For PostgreSQL instances, you can specify only one table.
      threads: Optional. The number of threads to use for parallel export.
    """

        class MysqlExportOptionsValue(_messages.Message):
            """Options for exporting from MySQL.

      Fields:
        masterData: Option to include SQL statement required to set up
          replication. If set to `1`, the dump file includes a CHANGE MASTER
          TO statement with the binary log coordinates, and --set-gtid-purged
          is set to ON. If set to `2`, the CHANGE MASTER TO statement is
          written as a SQL comment and has no effect. If set to any value
          other than `1`, --set-gtid-purged is set to OFF.
      """
            masterData = _messages.IntegerField(1, variant=_messages.Variant.INT32)
        mysqlExportOptions = _messages.MessageField('MysqlExportOptionsValue', 1)
        parallel = _messages.BooleanField(2)
        schemaOnly = _messages.BooleanField(3)
        tables = _messages.StringField(4, repeated=True)
        threads = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    bakExportOptions = _messages.MessageField('BakExportOptionsValue', 1)
    csvExportOptions = _messages.MessageField('CsvExportOptionsValue', 2)
    databases = _messages.StringField(3, repeated=True)
    fileType = _messages.EnumField('FileTypeValueValuesEnum', 4)
    kind = _messages.StringField(5)
    offload = _messages.BooleanField(6)
    sqlExportOptions = _messages.MessageField('SqlExportOptionsValue', 7)
    uri = _messages.StringField(8)