from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowV2beta1WebhookResponse(_messages.Message):
    """The response message for a webhook call. This response is validated by
  the Dialogflow server. If validation fails, an error will be returned in the
  QueryResult.diagnostic_info field. Setting JSON fields to an empty value
  with the wrong type is a common error. To avoid this error: - Use `""` for
  empty strings - Use `{}` or `null` for empty objects - Use `[]` or `null`
  for empty arrays For more information, see the [Protocol Buffers Language
  Guide](https://developers.google.com/protocol-buffers/docs/proto3#json).

  Messages:
    PayloadValue: Optional. This field can be used to pass custom data from
      your webhook to the integration or API caller. Arbitrary JSON objects
      are supported. When provided, Dialogflow uses this field to populate
      QueryResult.webhook_payload sent to the integration or API caller. This
      field is also used by the [Google Assistant
      integration](https://cloud.google.com/dialogflow/docs/integrations/aog)
      for rich response messages. See the format definition at [Google
      Assistant Dialogflow webhook format](https://developers.google.com/assis
      tant/actions/build/json/dialogflow-webhook-json)

  Fields:
    endInteraction: Optional. Indicates that this intent ends an interaction.
      Some integrations (e.g., Actions on Google or Dialogflow phone gateway)
      use this information to close interaction with an end user. Default is
      false.
    followupEventInput: Optional. Invokes the supplied events. When this field
      is set, Dialogflow ignores the `fulfillment_text`,
      `fulfillment_messages`, and `payload` fields.
    fulfillmentMessages: Optional. The rich response messages intended for the
      end-user. When provided, Dialogflow uses this field to populate
      QueryResult.fulfillment_messages sent to the integration or API caller.
    fulfillmentText: Optional. The text response message intended for the end-
      user. It is recommended to use `fulfillment_messages.text.text[0]`
      instead. When provided, Dialogflow uses this field to populate
      QueryResult.fulfillment_text sent to the integration or API caller.
    liveAgentHandoff: Indicates that a live agent should be brought in to
      handle the interaction with the user. In most cases, when you set this
      flag to true, you would also want to set end_interaction to true as
      well. Default is false.
    outputContexts: Optional. The collection of output contexts that will
      overwrite currently active contexts for the session and reset their
      lifespans. When provided, Dialogflow uses this field to populate
      QueryResult.output_contexts sent to the integration or API caller.
    payload: Optional. This field can be used to pass custom data from your
      webhook to the integration or API caller. Arbitrary JSON objects are
      supported. When provided, Dialogflow uses this field to populate
      QueryResult.webhook_payload sent to the integration or API caller. This
      field is also used by the [Google Assistant
      integration](https://cloud.google.com/dialogflow/docs/integrations/aog)
      for rich response messages. See the format definition at [Google
      Assistant Dialogflow webhook format](https://developers.google.com/assis
      tant/actions/build/json/dialogflow-webhook-json)
    sessionEntityTypes: Optional. Additional session entity types to replace
      or extend developer entity types with. The entity synonyms apply to all
      languages and persist for the session. Setting this data from a webhook
      overwrites the session entity types that have been set using
      `detectIntent`, `streamingDetectIntent` or SessionEntityType management
      methods.
    source: Optional. A custom field used to identify the webhook source.
      Arbitrary strings are supported. When provided, Dialogflow uses this
      field to populate QueryResult.webhook_source sent to the integration or
      API caller.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class PayloadValue(_messages.Message):
        """Optional. This field can be used to pass custom data from your webhook
    to the integration or API caller. Arbitrary JSON objects are supported.
    When provided, Dialogflow uses this field to populate
    QueryResult.webhook_payload sent to the integration or API caller. This
    field is also used by the [Google Assistant
    integration](https://cloud.google.com/dialogflow/docs/integrations/aog)
    for rich response messages. See the format definition at [Google Assistant
    Dialogflow webhook format](https://developers.google.com/assistant/actions
    /build/json/dialogflow-webhook-json)

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
    endInteraction = _messages.BooleanField(1)
    followupEventInput = _messages.MessageField('GoogleCloudDialogflowV2beta1EventInput', 2)
    fulfillmentMessages = _messages.MessageField('GoogleCloudDialogflowV2beta1IntentMessage', 3, repeated=True)
    fulfillmentText = _messages.StringField(4)
    liveAgentHandoff = _messages.BooleanField(5)
    outputContexts = _messages.MessageField('GoogleCloudDialogflowV2beta1Context', 6, repeated=True)
    payload = _messages.MessageField('PayloadValue', 7)
    sessionEntityTypes = _messages.MessageField('GoogleCloudDialogflowV2beta1SessionEntityType', 8, repeated=True)
    source = _messages.StringField(9)