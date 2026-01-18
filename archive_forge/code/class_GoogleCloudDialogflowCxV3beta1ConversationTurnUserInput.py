from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1ConversationTurnUserInput(_messages.Message):
    """The input from the human user.

  Messages:
    InjectedParametersValue: Parameters that need to be injected into the
      conversation during intent detection.

  Fields:
    enableSentimentAnalysis: Whether sentiment analysis is enabled.
    injectedParameters: Parameters that need to be injected into the
      conversation during intent detection.
    input: Supports text input, event input, dtmf input in the test case.
    isWebhookEnabled: If webhooks should be allowed to trigger in response to
      the user utterance. Often if parameters are injected, webhooks should
      not be enabled.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class InjectedParametersValue(_messages.Message):
        """Parameters that need to be injected into the conversation during
    intent detection.

    Messages:
      AdditionalProperty: An additional property for a InjectedParametersValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a InjectedParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    enableSentimentAnalysis = _messages.BooleanField(1)
    injectedParameters = _messages.MessageField('InjectedParametersValue', 2)
    input = _messages.MessageField('GoogleCloudDialogflowCxV3beta1QueryInput', 3)
    isWebhookEnabled = _messages.BooleanField(4)