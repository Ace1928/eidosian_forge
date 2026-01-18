from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
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