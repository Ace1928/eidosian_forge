from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleApiExprValue(_messages.Message):
    """Represents a CEL value. This is similar to `google.protobuf.Value`, but
  can represent CEL's full range of values.

  Enums:
    NullValueValueValuesEnum: Null value.

  Messages:
    ObjectValueValue: The proto message backing an object value.

  Fields:
    boolValue: Boolean value.
    bytesValue: Byte string value.
    doubleValue: Floating point value.
    enumValue: An enum value.
    int64Value: Signed integer value.
    listValue: List value.
    mapValue: Map value.
    nullValue: Null value.
    objectValue: The proto message backing an object value.
    stringValue: UTF-8 string value.
    typeValue: Type value.
    uint64Value: Unsigned integer value.
  """

    class NullValueValueValuesEnum(_messages.Enum):
        """Null value.

    Values:
      NULL_VALUE: Null value.
    """
        NULL_VALUE = 0

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ObjectValueValue(_messages.Message):
        """The proto message backing an object value.

    Messages:
      AdditionalProperty: An additional property for a ObjectValueValue
        object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ObjectValueValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    boolValue = _messages.BooleanField(1)
    bytesValue = _messages.BytesField(2)
    doubleValue = _messages.FloatField(3)
    enumValue = _messages.MessageField('GoogleApiExprEnumValue', 4)
    int64Value = _messages.IntegerField(5)
    listValue = _messages.MessageField('GoogleApiExprListValue', 6)
    mapValue = _messages.MessageField('GoogleApiExprMapValue', 7)
    nullValue = _messages.EnumField('NullValueValueValuesEnum', 8)
    objectValue = _messages.MessageField('ObjectValueValue', 9)
    stringValue = _messages.StringField(10)
    typeValue = _messages.StringField(11)
    uint64Value = _messages.IntegerField(12, variant=_messages.Variant.UINT64)