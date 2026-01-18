from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StandardSqlStructType(_messages.Message):
    """The representation of a SQL STRUCT type.

  Fields:
    fields: Fields within the struct.
  """
    fields = _messages.MessageField('StandardSqlField', 1, repeated=True)