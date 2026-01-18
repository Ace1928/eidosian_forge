from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class BetaSettingsValue(_messages.Message):
    """Metadata settings that are supplied to this version to enable beta
    runtime features.

    Messages:
      AdditionalProperty: An additional property for a BetaSettingsValue
        object.

    Fields:
      additionalProperties: Additional properties of type BetaSettingsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a BetaSettingsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
        key = _messages.StringField(1)
        value = _messages.StringField(2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)