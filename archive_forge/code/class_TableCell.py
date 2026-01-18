from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TableCell(_messages.Message):
    """A TableCell object.

  Fields:
    v: A extra_types.JsonValue attribute.
  """
    v = _messages.MessageField('extra_types.JsonValue', 1)