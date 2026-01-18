from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def CsvImportContext(sql_messages, uri, database, table, columns=None, user=None, quote=None, escape=None, fields_terminated_by=None, lines_terminated_by=None):
    """Generates the ImportContext for the given args, for importing from CSV.

  Args:
    sql_messages: module, The messages module that should be used.
    uri: The URI of the bucket to import from; the output of the 'uri' arg.
    database: The database to import into; the output of the '--database' flag.
    table: The table to import into; the output of the '--table' flag.
    columns: The CSV columns to import form; the output of the '--columns' flag.
    user: The Postgres user to import as; the output of the '--user' flag.
    quote: character in Hex. The quote character for CSV format; the output of
      the '--quote' flag.
    escape: character in Hex. The escape character for CSV format; the output of
      the '--escape' flag.
    fields_terminated_by: character in Hex. The fields delimiter character for
      CSV format; the output of the '--fields-terminated-by' flag.
    lines_terminated_by: character in Hex. The lines delimiter character for CSV
      format; the output of the '--lines-terminated-by' flag.

  Returns:
    ImportContext, for use in InstancesImportRequest.importContext.
  """
    return sql_messages.ImportContext(kind='sql#importContext', csvImportOptions=sql_messages.ImportContext.CsvImportOptionsValue(columns=columns or [], table=table, quoteCharacter=quote, escapeCharacter=escape, fieldsTerminatedBy=fields_terminated_by, linesTerminatedBy=lines_terminated_by), uri=uri, database=database, fileType=sql_messages.ImportContext.FileTypeValueValuesEnum.CSV, importUser=user)