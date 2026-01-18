from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ParamValue(_messages.Message):
    """Parameter value.

  Enums:
    TypeValueValuesEnum: Type of parameter.

  Fields:
    arrayVal: Value of the parameter if type is array.
    stringVal: Value of the parameter if type is string.
    type: Type of parameter.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Type of parameter.

    Values:
      TYPE_UNSPECIFIED: Default enum type; should not be used.
      STRING: Default
      ARRAY: Array type
    """
        TYPE_UNSPECIFIED = 0
        STRING = 1
        ARRAY = 2
    arrayVal = _messages.StringField(1, repeated=True)
    stringVal = _messages.StringField(2)
    type = _messages.EnumField('TypeValueValuesEnum', 3)