from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1WebhookRequestIntentInfo(_messages.Message):
    """Represents intent information communicated to the webhook.

  Messages:
    ParametersValue: Parameters identified as a result of intent matching.
      This is a map of the name of the identified parameter to the value of
      the parameter identified from the user's utterance. All parameters
      defined in the matched intent that are identified will be surfaced here.

  Fields:
    confidence: The confidence of the matched intent. Values range from 0.0
      (completely uncertain) to 1.0 (completely certain).
    displayName: Always present. The display name of the last matched intent.
    lastMatchedIntent: Always present. The unique identifier of the last
      matched intent. Format: `projects//locations//agents//intents/`.
    parameters: Parameters identified as a result of intent matching. This is
      a map of the name of the identified parameter to the value of the
      parameter identified from the user's utterance. All parameters defined
      in the matched intent that are identified will be surfaced here.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """Parameters identified as a result of intent matching. This is a map of
    the name of the identified parameter to the value of the parameter
    identified from the user's utterance. All parameters defined in the
    matched intent that are identified will be surfaced here.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Additional properties of type ParametersValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A GoogleCloudDialogflowCxV3beta1WebhookRequestIntentInfoIntentP
          arameterValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('GoogleCloudDialogflowCxV3beta1WebhookRequestIntentInfoIntentParameterValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    confidence = _messages.FloatField(1, variant=_messages.Variant.FLOAT)
    displayName = _messages.StringField(2)
    lastMatchedIntent = _messages.StringField(3)
    parameters = _messages.MessageField('ParametersValue', 4)