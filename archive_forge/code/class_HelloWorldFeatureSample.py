from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HelloWorldFeatureSample(_messages.Message):
    """Represents message used in feature e2e create/mutate testing.

  Enums:
    ThirdValueValuesEnum:

  Messages:
    FifthValue: A FifthValue object.
    NinthValue: Map field.

  Fields:
    eighth: Repeated field.
    fifth: A FifthValue attribute.
    first: Singular scaler field.
    fourth: Singular Message fields.
    ninth: Map field.
    second: Singular scaler field.
    seventh: A string attribute.
    sixth: A string attribute.
    third: A ThirdValueValuesEnum attribute.
  """

    class ThirdValueValuesEnum(_messages.Enum):
        """ThirdValueValuesEnum enum type.

    Values:
      BAR_UNSPECIFIED: <no description>
      FIRST: <no description>
      SECOND: <no description>
    """
        BAR_UNSPECIFIED = 0
        FIRST = 1
        SECOND = 2

    @encoding.MapUnrecognizedFields('additionalProperties')
    class FifthValue(_messages.Message):
        """A FifthValue object.

    Messages:
      AdditionalProperty: An additional property for a FifthValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a FifthValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class NinthValue(_messages.Message):
        """Map field.

    Messages:
      AdditionalProperty: An additional property for a NinthValue object.

    Fields:
      additionalProperties: Additional properties of type NinthValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a NinthValue object.

      Fields:
        key: Name of the additional property.
        value: A HelloWorldFooBar attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('HelloWorldFooBar', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    eighth = _messages.MessageField('HelloWorldFooBar', 1, repeated=True)
    fifth = _messages.MessageField('FifthValue', 2)
    first = _messages.StringField(3)
    fourth = _messages.StringField(4)
    ninth = _messages.MessageField('NinthValue', 5)
    second = _messages.IntegerField(6, variant=_messages.Variant.INT32)
    seventh = _messages.StringField(7)
    sixth = _messages.IntegerField(8)
    third = _messages.EnumField('ThirdValueValuesEnum', 9)