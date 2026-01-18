from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class SynthesizeSpeechConfigsValue(_messages.Message):
    """Optional. Configuration of how speech should be synthesized, mapping
    from language
    (https://cloud.google.com/dialogflow/docs/reference/language) to
    SynthesizeSpeechConfig.

    Messages:
      AdditionalProperty: An additional property for a
        SynthesizeSpeechConfigsValue object.

    Fields:
      additionalProperties: Additional properties of type
        SynthesizeSpeechConfigsValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a SynthesizeSpeechConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDialogflowV2SynthesizeSpeechConfig attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('GoogleCloudDialogflowV2SynthesizeSpeechConfig', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)