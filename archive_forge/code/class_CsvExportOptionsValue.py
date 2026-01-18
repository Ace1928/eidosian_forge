from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
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