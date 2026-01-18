from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
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