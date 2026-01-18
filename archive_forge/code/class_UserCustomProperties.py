from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class UserCustomProperties(_messages.Message):
    """JSON template for a set of custom properties (i.e.

  all fields in a
  particular schema)

  Messages:
    AdditionalProperty: An additional property for a UserCustomProperties
      object.

  Fields:
    additionalProperties: Additional properties of type UserCustomProperties
  """

    class AdditionalProperty(_messages.Message):
        """An additional property for a UserCustomProperties object.

    Fields:
      key: Name of the additional property.
      value: A extra_types.JsonValue attribute.
    """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)