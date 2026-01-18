from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SqlServerSchema(_messages.Message):
    """SQLServer schema.

  Fields:
    schema: Schema name.
    tables: Tables in the schema.
  """
    schema = _messages.StringField(1)
    tables = _messages.MessageField('SqlServerTable', 2, repeated=True)