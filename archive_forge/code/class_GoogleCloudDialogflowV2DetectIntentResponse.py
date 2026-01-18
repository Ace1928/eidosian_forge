from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2DetectIntentResponse(_messages.Message):
    """The message returned from the DetectIntent method.

  Fields:
    outputAudio: The audio data bytes encoded as specified in the request.
      Note: The output audio is generated based on the values of default
      platform text responses found in the `query_result.fulfillment_messages`
      field. If multiple default text responses exist, they will be
      concatenated when generating audio. If no default platform text
      responses exist, the generated audio content will be empty. In some
      scenarios, multiple output audio fields may be present in the response
      structure. In these cases, only the top-most-level audio output has
      content.
    outputAudioConfig: The config used by the speech synthesizer to generate
      the output audio.
    queryResult: The selected results of the conversational query or event
      processing. See `alternative_query_results` for additional potential
      results.
    responseId: The unique identifier of the response. It can be used to
      locate a response in the training example set or for reporting issues.
    webhookStatus: Specifies the status of the webhook request.
  """
    outputAudio = _messages.BytesField(1)
    outputAudioConfig = _messages.MessageField('GoogleCloudDialogflowV2OutputAudioConfig', 2)
    queryResult = _messages.MessageField('GoogleCloudDialogflowV2QueryResult', 3)
    responseId = _messages.StringField(4)
    webhookStatus = _messages.MessageField('GoogleRpcStatus', 5)