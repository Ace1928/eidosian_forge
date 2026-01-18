from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class ChannelsValue(_messages.Message):
    """Output only. Mapping from release channel to channel config.

    Messages:
      AdditionalProperty: An additional property for a ChannelsValue object.

    Fields:
      additionalProperties: Additional properties of type ChannelsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ChannelsValue object.

      Fields:
        key: Name of the additional property.
        value: A ChannelConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('ChannelConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)