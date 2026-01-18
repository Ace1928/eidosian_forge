from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ArrayValue(_messages.Message):
    """An array value.

  Fields:
    values: Values in the array.
  """
    values = _messages.MessageField('Value', 1, repeated=True)