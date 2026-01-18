from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTableImportRowsRequest(_messages.Message):
    """A FusiontablesTableImportRowsRequest object.

  Fields:
    delimiter: The delimiter used to separate cell values. This can only
      consist of a single character. Default is ','.
    encoding: The encoding of the content. Default is UTF-8. Use 'auto-detect'
      if you are unsure of the encoding.
    endLine: The index of the last line from which to start importing,
      exclusive. Thus, the number of imported lines is endLine - startLine. If
      this parameter is not provided, the file will be imported until the last
      line of the file. If endLine is negative, then the imported content will
      exclude the last endLine lines. That is, if endline is negative, no line
      will be imported whose index is greater than N + endLine where N is the
      number of lines in the file, and the number of imported lines will be N
      + endLine - startLine.
    isStrict: Whether the CSV must have the same number of values for each
      row. If false, rows with fewer values will be padded with empty values.
      Default is true.
    startLine: The index of the first line from which to start importing,
      inclusive. Default is 0.
    tableId: The table into which new rows are being imported.
  """
    delimiter = _messages.StringField(1)
    encoding = _messages.StringField(2)
    endLine = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    isStrict = _messages.BooleanField(4)
    startLine = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    tableId = _messages.StringField(6, required=True)