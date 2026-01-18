from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StandardSqlField(_messages.Message):
    """A field or a column.

  Fields:
    name: Optional. The name of this field. Can be absent for struct fields.
    type: Optional. The type of this parameter. Absent if not explicitly
      specified (e.g., CREATE FUNCTION statement can omit the return type; in
      this case the output parameter does not have this "type" field).
  """
    name = _messages.StringField(1)
    type = _messages.MessageField('StandardSqlDataType', 2)