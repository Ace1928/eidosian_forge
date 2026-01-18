from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResultValue(_messages.Message):
    """ResultValue holds different types of data for a single result.

  Enums:
    TypeValueValuesEnum: Output only. The type of data that the result holds.

  Messages:
    ObjectValValue: Value of the result if type is object.

  Fields:
    arrayVal: Value of the result if type is array.
    objectVal: Value of the result if type is object.
    stringVal: Value of the result if type is string.
    type: Output only. The type of data that the result holds.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Output only. The type of data that the result holds.

    Values:
      TYPE_UNSPECIFIED: Default enum type; should not be used.
      STRING: Default
      ARRAY: Array type
      OBJECT: Object type
    """
        TYPE_UNSPECIFIED = 0
        STRING = 1
        ARRAY = 2
        OBJECT = 3

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ObjectValValue(_messages.Message):
        """Value of the result if type is object.

    Messages:
      AdditionalProperty: An additional property for a ObjectValValue object.

    Fields:
      additionalProperties: Additional properties of type ObjectValValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ObjectValValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    arrayVal = _messages.StringField(1, repeated=True)
    objectVal = _messages.MessageField('ObjectValValue', 2)
    stringVal = _messages.StringField(3)
    type = _messages.EnumField('TypeValueValuesEnum', 4)