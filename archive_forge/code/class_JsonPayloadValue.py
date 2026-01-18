from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class JsonPayloadValue(_messages.Message):
    """The log entry payload, represented as a structure that is expressed as
    a JSON object.

    Messages:
      AdditionalProperty: An additional property for a JsonPayloadValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a JsonPayloadValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('extra_types.JsonValue', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)