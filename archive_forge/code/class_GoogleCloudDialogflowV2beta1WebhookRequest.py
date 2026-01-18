from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1WebhookRequest(_messages.Message):
    """The request message for a webhook call.

  Fields:
    alternativeQueryResults: Alternative query results from KnowledgeService.
    originalDetectIntentRequest: Optional. The contents of the original
      request that was passed to `[Streaming]DetectIntent` call.
    queryResult: The result of the conversational query or event processing.
      Contains the same value as
      `[Streaming]DetectIntentResponse.query_result`.
    responseId: The unique identifier of the response. Contains the same value
      as `[Streaming]DetectIntentResponse.response_id`.
    session: The unique identifier of detectIntent request session. Can be
      used to identify end-user inside webhook implementation. Supported
      formats: - `projects//agent/sessions/, -
      `projects//locations//agent/sessions/`, -
      `projects//agent/environments//users//sessions/`, -
      `projects//locations//agent/environments//users//sessions/`,
  """
    alternativeQueryResults = _messages.MessageField('GoogleCloudDialogflowV2beta1QueryResult', 1, repeated=True)
    originalDetectIntentRequest = _messages.MessageField('GoogleCloudDialogflowV2beta1OriginalDetectIntentRequest', 2)
    queryResult = _messages.MessageField('GoogleCloudDialogflowV2beta1QueryResult', 3)
    responseId = _messages.StringField(4)
    session = _messages.StringField(5)