from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class DefinitionsValue(_messages.Message):
    """A DefinitionsValue object.

    Messages:
      AdditionalProperty: An additional property for a DefinitionsValue
        object.

    Fields:
      additionalProperties: Additional properties of type DefinitionsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a DefinitionsValue object.

      Fields:
        key: Name of the additional property.
        value: A JSONSchemaProps attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('JSONSchemaProps', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)