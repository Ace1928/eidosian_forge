from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprTypeMapType(_messages.Message):
    """Map type with parameterized key and value types, e.g. `map`.

  Fields:
    keyType: The type of the key.
    valueType: The type of the value.
  """
    keyType = _messages.MessageField('GoogleApiExprType', 1)
    valueType = _messages.MessageField('GoogleApiExprType', 2)