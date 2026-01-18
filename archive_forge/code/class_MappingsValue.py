from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class MappingsValue(_messages.Message):
    """The key-value pairs in the map

    Messages:
      AdditionalProperty: An additional property for a MappingsValue object.

    Fields:
      additionalProperties: Additional properties of type MappingsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a MappingsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)