from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3beta1Webhook(_messages.Message):
    """Webhooks host the developer's business logic. During a session, webhooks
  allow the developer to use the data extracted by Dialogflow's natural
  language processing to generate dynamic responses, validate collected data,
  or trigger actions on the backend.

  Fields:
    disabled: Indicates whether the webhook is disabled.
    displayName: Required. The human-readable name of the webhook, unique
      within the agent.
    genericWebService: Configuration for a generic web service.
    name: The unique identifier of the webhook. Required for the
      Webhooks.UpdateWebhook method. Webhooks.CreateWebhook populates the name
      automatically. Format: `projects//locations//agents//webhooks/`.
    serviceDirectory: Configuration for a [Service
      Directory](https://cloud.google.com/service-directory) service.
    timeout: Webhook execution timeout. Execution is considered failed if
      Dialogflow doesn't receive a response from webhook at the end of the
      timeout period. Defaults to 5 seconds, maximum allowed timeout is 30
      seconds.
  """
    disabled = _messages.BooleanField(1)
    displayName = _messages.StringField(2)
    genericWebService = _messages.MessageField('GoogleCloudDialogflowCxV3beta1WebhookGenericWebService', 3)
    name = _messages.StringField(4)
    serviceDirectory = _messages.MessageField('GoogleCloudDialogflowCxV3beta1WebhookServiceDirectoryConfig', 5)
    timeout = _messages.StringField(6)