from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerTable(_messages.Message):
    """SQLServer table.

  Fields:
    columns: SQLServer columns in the schema. When unspecified as part of
      include/exclude objects, includes/excludes everything.
    table: Table name.
  """
    columns = _messages.MessageField('SqlServerColumn', 1, repeated=True)
    table = _messages.StringField(2)