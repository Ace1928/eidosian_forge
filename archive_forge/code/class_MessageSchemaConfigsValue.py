from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class MessageSchemaConfigsValue(_messages.Message):
    """Map from each HL7v2 message type and trigger event pair, such as
    ADT_A04, to its schema configuration root group.

    Messages:
      AdditionalProperty: An additional property for a
        MessageSchemaConfigsValue object.

    Fields:
      additionalProperties: Additional properties of type
        MessageSchemaConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MessageSchemaConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A SchemaGroup attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('SchemaGroup', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)