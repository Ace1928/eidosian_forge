from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class OracleSchema(_messages.Message):
    """Oracle schema.

  Fields:
    oracleTables: Tables in the schema.
    schema: Schema name.
  """
    oracleTables = _messages.MessageField('OracleTable', 1, repeated=True)
    schema = _messages.StringField(2)