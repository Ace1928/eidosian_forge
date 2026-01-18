from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1Fulfillment(_messages.Message):
    """A fulfillment can do one or more of the following actions at the same
  time: * Generate rich message responses. * Set parameter values. * Call the
  webhook. Fulfillments can be called at various stages in the Page or Form
  lifecycle. For example, when a DetectIntentRequest drives a session to enter
  a new page, the page's entry fulfillment can add a static response to the
  QueryResult in the returning DetectIntentResponse, call the webhook (for
  example, to load user data from a database), or both.

  Fields:
    advancedSettings: Hierarchical advanced settings for this fulfillment. The
      settings exposed at the lower level overrides the settings exposed at
      the higher level.
    conditionalCases: Conditional cases for this fulfillment.
    enableGenerativeFallback: If the flag is true, the agent will utilize LLM
      to generate a text response. If LLM generation fails, the defined
      responses in the fulfillment will be respected. This flag is only useful
      for fulfillments associated with no-match event handlers.
    messages: The list of rich message responses to present to the user.
    returnPartialResponses: Whether Dialogflow should return currently queued
      fulfillment response messages in streaming APIs. If a webhook is
      specified, it happens before Dialogflow invokes webhook. Warning: 1)
      This flag only affects streaming API. Responses are still queued and
      returned once in non-streaming API. 2) The flag can be enabled in any
      fulfillment but only the first 3 partial responses will be returned. You
      may only want to apply it to fulfillments that have slow webhooks.
    setParameterActions: Set parameter values before executing the webhook.
    tag: The value of this field will be populated in the WebhookRequest
      `fulfillmentInfo.tag` field by Dialogflow when the associated webhook is
      called. The tag is typically used by the webhook service to identify
      which fulfillment is being called, but it could be used for other
      purposes. This field is required if `webhook` is specified.
    webhook: The webhook to call. Format:
      `projects//locations//agents//webhooks/`.
  """
    advancedSettings = _messages.MessageField('GoogleCloudDialogflowCxV3beta1AdvancedSettings', 1)
    conditionalCases = _messages.MessageField('GoogleCloudDialogflowCxV3beta1FulfillmentConditionalCases', 2, repeated=True)
    enableGenerativeFallback = _messages.BooleanField(3)
    messages = _messages.MessageField('GoogleCloudDialogflowCxV3beta1ResponseMessage', 4, repeated=True)
    returnPartialResponses = _messages.BooleanField(5)
    setParameterActions = _messages.MessageField('GoogleCloudDialogflowCxV3beta1FulfillmentSetParameterAction', 6, repeated=True)
    tag = _messages.StringField(7)
    webhook = _messages.StringField(8)