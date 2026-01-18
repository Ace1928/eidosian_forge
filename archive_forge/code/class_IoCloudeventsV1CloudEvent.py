from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IoCloudeventsV1CloudEvent(_messages.Message):
    """-- CloudEvent Context Attributes

  Messages:
    AttributesValue: Optional & Extension Attributes
    ProtoDataValue: A ProtoDataValue object.

  Fields:
    attributes: Optional & Extension Attributes
    binaryData: A byte attribute.
    id: Required Attributes
    protoData: A ProtoDataValue attribute.
    source: URI-reference
    specVersion: A string attribute.
    textData: A string attribute.
    type: A string attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class AttributesValue(_messages.Message):
        """Optional & Extension Attributes

    Messages:
      AdditionalProperty: An additional property for a AttributesValue object.

    Fields:
      additionalProperties: Additional properties of type AttributesValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a AttributesValue object.

      Fields:
        key: Name of the additional property.
        value: A IoCloudeventsV1CloudEventCloudEventAttributeValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('IoCloudeventsV1CloudEventCloudEventAttributeValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ProtoDataValue(_messages.Message):
        """A ProtoDataValue object.

    Messages:
      AdditionalProperty: An additional property for a ProtoDataValue object.

    Fields:
      additionalProperties: Properties of the object. Contains field @type
        with type URL.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ProtoDataValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    attributes = _messages.MessageField('AttributesValue', 1)
    binaryData = _messages.BytesField(2)
    id = _messages.StringField(3)
    protoData = _messages.MessageField('ProtoDataValue', 4)
    source = _messages.StringField(5)
    specVersion = _messages.StringField(6)
    textData = _messages.StringField(7)
    type = _messages.StringField(8)