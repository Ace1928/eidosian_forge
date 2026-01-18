from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2AnalyzeContentRequest(_messages.Message):
    """The request message for Participants.AnalyzeContent.

  Messages:
    CxParametersValue: Additional parameters to be put into Dialogflow CX
      session parameters. To remove a parameter from the session, clients
      should explicitly set the parameter value to null. Note: this field
      should only be used if you are connecting to a Dialogflow CX agent.

  Fields:
    assistQueryParams: Parameters for a human assist query.
    cxParameters: Additional parameters to be put into Dialogflow CX session
      parameters. To remove a parameter from the session, clients should
      explicitly set the parameter value to null. Note: this field should only
      be used if you are connecting to a Dialogflow CX agent.
    eventInput: An input event to send to Dialogflow.
    queryParams: Parameters for a Dialogflow virtual-agent query.
    replyAudioConfig: Speech synthesis configuration. The speech synthesis
      settings for a virtual agent that may be configured for the associated
      conversation profile are not used when calling AnalyzeContent. If this
      configuration is not supplied, speech synthesis is disabled.
    requestId: A unique identifier for this request. Restricted to 36 ASCII
      characters. A random UUID is recommended. This request is only
      idempotent if a `request_id` is provided.
    suggestionInput: An input representing the selection of a suggestion.
    textInput: The natural language text to be processed.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class CxParametersValue(_messages.Message):
        """Additional parameters to be put into Dialogflow CX session parameters.
    To remove a parameter from the session, clients should explicitly set the
    parameter value to null. Note: this field should only be used if you are
    connecting to a Dialogflow CX agent.

    Messages:
      AdditionalProperty: An additional property for a CxParametersValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a CxParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    assistQueryParams = _messages.MessageField('GoogleCloudDialogflowV2AssistQueryParameters', 1)
    cxParameters = _messages.MessageField('CxParametersValue', 2)
    eventInput = _messages.MessageField('GoogleCloudDialogflowV2EventInput', 3)
    queryParams = _messages.MessageField('GoogleCloudDialogflowV2QueryParameters', 4)
    replyAudioConfig = _messages.MessageField('GoogleCloudDialogflowV2OutputAudioConfig', 5)
    requestId = _messages.StringField(6)
    suggestionInput = _messages.MessageField('GoogleCloudDialogflowV2SuggestionInput', 7)
    textInput = _messages.MessageField('GoogleCloudDialogflowV2TextInput', 8)