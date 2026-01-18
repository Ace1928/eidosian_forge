from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1QueryResult(_messages.Message):
    """Represents the result of conversational query or event processing.

  Messages:
    DiagnosticInfoValue: Free-form diagnostic information for the associated
      detect intent request. The fields of this data can change without
      notice, so you should not write code that depends on its structure. The
      data may contain: - webhook call latency - webhook errors
    ParametersValue: The collection of extracted parameters. Depending on your
      protocol or client library language, this is a map, associative array,
      symbol table, dictionary, or JSON object composed of a collection of
      (MapKey, MapValue) pairs: * MapKey type: string * MapKey value:
      parameter name * MapValue type: If parameter's entity type is a
      composite entity then use map, otherwise, depending on the parameter
      value type, it could be one of string, number, boolean, null, list or
      map. * MapValue value: If parameter's entity type is a composite entity
      then use map from composite entity property names to property values,
      otherwise, use parameter value.
    WebhookPayloadValue: If the query was fulfilled by a webhook call, this
      field is set to the value of the `payload` field returned in the webhook
      response.

  Fields:
    action: The action name from the matched intent.
    allRequiredParamsPresent: This field is set to: - `false` if the matched
      intent has required parameters and not all of the required parameter
      values have been collected. - `true` if all required parameter values
      have been collected, or if the matched intent doesn't contain any
      required parameters.
    cancelsSlotFilling: Indicates whether the conversational query triggers a
      cancellation for slot filling. For more information, see the [cancel
      slot filling
      documentation](https://cloud.google.com/dialogflow/es/docs/intents-
      actions-parameters#cancel).
    diagnosticInfo: Free-form diagnostic information for the associated detect
      intent request. The fields of this data can change without notice, so
      you should not write code that depends on its structure. The data may
      contain: - webhook call latency - webhook errors
    fulfillmentMessages: The collection of rich messages to present to the
      user.
    fulfillmentText: The text to be pronounced to the user or shown on the
      screen. Note: This is a legacy field, `fulfillment_messages` should be
      preferred.
    intent: The intent that matched the conversational query. Some, not all
      fields are filled in this message, including but not limited to: `name`,
      `display_name`, `end_interaction` and `is_fallback`.
    intentDetectionConfidence: The intent detection confidence. Values range
      from 0.0 (completely uncertain) to 1.0 (completely certain). This value
      is for informational purpose only and is only used to help match the
      best intent within the classification threshold. This value may change
      for the same end-user expression at any time due to a model retraining
      or change in implementation. If there are `multiple knowledge_answers`
      messages, this value is set to the greatest
      `knowledgeAnswers.match_confidence` value in the list.
    knowledgeAnswers: The result from Knowledge Connector (if any), ordered by
      decreasing `KnowledgeAnswers.match_confidence`.
    languageCode: The language that was triggered during intent detection. See
      [Language
      Support](https://cloud.google.com/dialogflow/docs/reference/language)
      for a list of the currently supported language codes.
    outputContexts: The collection of output contexts. If applicable,
      `output_contexts.parameters` contains entries with name `.original`
      containing the original parameter values before the query.
    parameters: The collection of extracted parameters. Depending on your
      protocol or client library language, this is a map, associative array,
      symbol table, dictionary, or JSON object composed of a collection of
      (MapKey, MapValue) pairs: * MapKey type: string * MapKey value:
      parameter name * MapValue type: If parameter's entity type is a
      composite entity then use map, otherwise, depending on the parameter
      value type, it could be one of string, number, boolean, null, list or
      map. * MapValue value: If parameter's entity type is a composite entity
      then use map from composite entity property names to property values,
      otherwise, use parameter value.
    queryText: The original conversational query text: - If natural language
      text was provided as input, `query_text` contains a copy of the input. -
      If natural language speech audio was provided as input, `query_text`
      contains the speech recognition result. If speech recognizer produced
      multiple alternatives, a particular one is picked. - If automatic spell
      correction is enabled, `query_text` will contain the corrected user
      input.
    sentimentAnalysisResult: The sentiment analysis result, which depends on
      the `sentiment_analysis_request_config` specified in the request.
    speechRecognitionConfidence: The Speech recognition confidence between 0.0
      and 1.0. A higher number indicates an estimated greater likelihood that
      the recognized words are correct. The default of 0.0 is a sentinel value
      indicating that confidence was not set. This field is not guaranteed to
      be accurate or set. In particular this field isn't set for
      StreamingDetectIntent since the streaming endpoint has separate
      confidence estimates per portion of the audio in
      StreamingRecognitionResult.
    webhookPayload: If the query was fulfilled by a webhook call, this field
      is set to the value of the `payload` field returned in the webhook
      response.
    webhookSource: If the query was fulfilled by a webhook call, this field is
      set to the value of the `source` field returned in the webhook response.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class DiagnosticInfoValue(_messages.Message):
        """Free-form diagnostic information for the associated detect intent
    request. The fields of this data can change without notice, so you should
    not write code that depends on its structure. The data may contain: -
    webhook call latency - webhook errors

    Messages:
      AdditionalProperty: An additional property for a DiagnosticInfoValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a DiagnosticInfoValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ParametersValue(_messages.Message):
        """The collection of extracted parameters. Depending on your protocol or
    client library language, this is a map, associative array, symbol table,
    dictionary, or JSON object composed of a collection of (MapKey, MapValue)
    pairs: * MapKey type: string * MapKey value: parameter name * MapValue
    type: If parameter's entity type is a composite entity then use map,
    otherwise, depending on the parameter value type, it could be one of
    string, number, boolean, null, list or map. * MapValue value: If
    parameter's entity type is a composite entity then use map from composite
    entity property names to property values, otherwise, use parameter value.

    Messages:
      AdditionalProperty: An additional property for a ParametersValue object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ParametersValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)

    @encoding.MapUnrecognizedFields('additionalProperties')
    class WebhookPayloadValue(_messages.Message):
        """If the query was fulfilled by a webhook call, this field is set to the
    value of the `payload` field returned in the webhook response.

    Messages:
      AdditionalProperty: An additional property for a WebhookPayloadValue
        object.

    Fields:
      additionalProperties: Properties of the object.
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a WebhookPayloadValue object.

      Fields:
        key: Name of the additional property.
        value: A extra_types.JsonValue attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('extra_types.JsonValue', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    action = _messages.StringField(1)
    allRequiredParamsPresent = _messages.BooleanField(2)
    cancelsSlotFilling = _messages.BooleanField(3)
    diagnosticInfo = _messages.MessageField('DiagnosticInfoValue', 4)
    fulfillmentMessages = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessage', 5, repeated=True)
    fulfillmentText = _messages.StringField(6)
    intent = _messages.MessageField('GoogleCloudDialogflowV2beta1Intent', 7)
    intentDetectionConfidence = _messages.FloatField(8, variant=_messages.Variant.FLOAT)
    knowledgeAnswers = _messages.MessageField('GoogleCloudDialogflowV2beta1KnowledgeAnswers', 9)
    languageCode = _messages.StringField(10)
    outputContexts = _messages.MessageField('GoogleCloudDialogflowV2beta1Context', 11, repeated=True)
    parameters = _messages.MessageField('ParametersValue', 12)
    queryText = _messages.StringField(13)
    sentimentAnalysisResult = _messages.MessageField('GoogleCloudDialogflowV2beta1SentimentAnalysisResult', 14)
    speechRecognitionConfidence = _messages.FloatField(15, variant=_messages.Variant.FLOAT)
    webhookPayload = _messages.MessageField('WebhookPayloadValue', 16)
    webhookSource = _messages.StringField(17)