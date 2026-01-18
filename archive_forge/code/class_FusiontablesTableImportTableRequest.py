from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import extra_types
class FusiontablesTableImportTableRequest(_messages.Message):
    """A FusiontablesTableImportTableRequest object.

  Fields:
    delimiter: The delimiter used to separate cell values. This can only
      consist of a single character. Default is ','.
    encoding: The encoding of the content. Default is UTF-8. Use 'auto-detect'
      if you are unsure of the encoding.
    name: The name to be assigned to the new table.
  """
    delimiter = _messages.StringField(1)
    encoding = _messages.StringField(2)
    name = _messages.StringField(3, required=True)