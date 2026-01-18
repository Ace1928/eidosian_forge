from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplexFieldAllowedValues(_messages.Message):
    """A ComplexFieldAllowedValues object.

  Fields:
    values: A extra_types.JsonValue attribute.
  """
    values = _messages.MessageField('extra_types.JsonValue', 1, repeated=True)