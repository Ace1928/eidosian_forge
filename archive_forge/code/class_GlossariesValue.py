from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class GlossariesValue(_messages.Message):
    """Optional. Glossaries to be applied for translation. It's keyed by
    target language code.

    Messages:
      AdditionalProperty: An additional property for a GlossariesValue object.

    Fields:
      additionalProperties: Additional properties of type GlossariesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a GlossariesValue object.

      Fields:
        key: Name of the additional property.
        value: A TranslateTextGlossaryConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('TranslateTextGlossaryConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)