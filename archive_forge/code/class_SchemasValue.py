from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class SchemasValue(_messages.Message):
    """The schemas for this API.

    Messages:
      AdditionalProperty: An additional property for a SchemasValue object.

    Fields:
      additionalProperties: An individual schema description.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SchemasValue object.

      Fields:
        key: Name of the additional property.
        value: A JsonSchema attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('JsonSchema', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)