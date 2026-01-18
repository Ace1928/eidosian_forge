from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StructType(_messages.Message):
    """`StructType` defines the fields of a STRUCT type.

  Fields:
    fields: The list of fields that make up this struct. Order is significant,
      because values of this struct type are represented as lists, where the
      order of field values matches the order of fields in the StructType. In
      turn, the order of fields matches the order of columns in a read
      request, or the order of fields in the `SELECT` clause of a query.
  """
    fields = _messages.MessageField('Field', 1, repeated=True)