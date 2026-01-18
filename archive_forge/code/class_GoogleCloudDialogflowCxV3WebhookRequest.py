from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3WebhookRequest(_messages.Message):
    """The request message for a webhook call. The request is sent as a JSON
  object and the field names will be presented in camel cases. You may see
  undocumented fields in an actual request. These fields are used internally
  by Dialogflow and should be ignored.

  Messages:
    PayloadValue: Custom data set in QueryParameters.payload.

  Fields:
    detectIntentResponseId: Always present. The unique identifier of the
      DetectIntentResponse that will be returned to the API caller.
    dtmfDigits: If DTMF was provided as input, this field will contain the
      DTMF digits.
    fulfillmentInfo: Always present. Information about the fulfillment that
      triggered this webhook call.
    intentInfo: Information about the last matched intent.
    languageCode: The language code specified in the original request.
    messages: The list of rich message responses to present to the user.
      Webhook can choose to append or replace this list in
      WebhookResponse.fulfillment_response;
    pageInfo: Information about page status.
    payload: Custom data set in QueryParameters.payload.
    sentimentAnalysisResult: The sentiment analysis result of the current user
      request. The field is filled when sentiment analysis is configured to be
      enabled for the request.
    sessionInfo: Information about session status.
    text: If natural language text was provided as input, this field will
      contain a copy of the text.
    transcript: If natural language speech audio was provided as input, this
      field will contain the transcript for the audio.
    triggerEvent: If an event was provided as input, this field will contain
      the name of the event.
    triggerIntent: If an intent was provided as input, this field will contain
      a copy of the intent identifier. Format:
      `projects//locations//agents//intents/`.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PayloadValue(_messages.Message):
        """Custom data set in QueryParameters.payload.

    Messages:
      AdditionalProperty: An additional property for a PayloadValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a PayloadValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    detectIntentResponseId = _messages.StringField(1)
    dtmfDigits = _messages.StringField(2)
    fulfillmentInfo = _messages.MessageField('GoogleCloudDialogflowCxV3WebhookRequestFulfillmentInfo', 3)
    intentInfo = _messages.MessageField('GoogleCloudDialogflowCxV3WebhookRequestIntentInfo', 4)
    languageCode = _messages.StringField(5)
    messages = _messages.MessageField('GoogleCloudDialogflowCxV3ResponseMessage', 6, repeated=True)
    pageInfo = _messages.MessageField('GoogleCloudDialogflowCxV3PageInfo', 7)
    payload = _messages.MessageField('PayloadValue', 8)
    sentimentAnalysisResult = _messages.MessageField('GoogleCloudDialogflowCxV3WebhookRequestSentimentAnalysisResult', 9)
    sessionInfo = _messages.MessageField('GoogleCloudDialogflowCxV3SessionInfo', 10)
    text = _messages.StringField(11)
    transcript = _messages.StringField(12)
    triggerEvent = _messages.StringField(13)
    triggerIntent = _messages.StringField(14)