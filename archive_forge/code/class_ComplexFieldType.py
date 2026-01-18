from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComplexFieldType(_messages.Message):
    """A ComplexFieldType object.

  Fields:
    allowedValues: If present, metadata values of ComplexFieldType must be
      within the constraints of allowedValues.
    required: If true, a value of this complex type must contain this field.
    type: Required. Type for this field. The type can be one of: - Primitive
      types ("string", "number", "bool") - Custom complex type. Format:
      "p/p/l/l/complexTypes/*" - Collections of the above - list(), dict()
  """
    allowedValues = _messages.MessageField('ComplexFieldAllowedValues', 1)
    required = _messages.BooleanField(2)
    type = _messages.StringField(3)