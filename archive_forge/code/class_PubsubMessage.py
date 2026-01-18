from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PubsubMessage(_messages.Message):
    """A message data and its attributes. The message payload must not be empty;
  it must contain either a non-empty data field, or at least one attribute.

  Messages:
    AttributesValue: Optional attributes for this message.

  Fields:
    attributes: Optional attributes for this message.
    data: The message payload.
    messageId: ID of this message, assigned by the server when the message is
      published. Guaranteed to be unique within the topic. This value may be
      read by a subscriber that receives a `PubsubMessage` via a `Pull` call
      or a push delivery. It must not be populated by the publisher in a
      `Publish` call.
    publishTime: The time at which the message was published, populated by the
      server when it receives the `Publish` call. It must not be populated by
      the publisher in a `Publish` call.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributesValue(_messages.Message):
        """Optional attributes for this message.

    Messages:
      AdditionalProperty: An additional property for a AttributesValue object.

    Fields:
      additionalProperties: Additional properties of type AttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attributes = _messages.MessageField('AttributesValue', 1)
    data = _messages.BytesField(2)
    messageId = _messages.StringField(3)
    publishTime = _messages.StringField(4)