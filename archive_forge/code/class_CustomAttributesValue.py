from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class CustomAttributesValue(_messages.Message):
    """An optional list of additional attributes to attach to each Cloud
    PubSub message published for this notification subscription.

    Messages:
      AdditionalProperty: An additional property for a CustomAttributesValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        CustomAttributesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a CustomAttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)