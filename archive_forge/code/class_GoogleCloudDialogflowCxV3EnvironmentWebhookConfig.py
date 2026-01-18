from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDialogflowCxV3EnvironmentWebhookConfig(_messages.Message):
    """Configuration for webhooks.

  Fields:
    webhookOverrides: The list of webhooks to override for the agent
      environment. The webhook must exist in the agent. You can override
      fields in `generic_web_service` and `service_directory`.
  """
    webhookOverrides = _messages.MessageField('GoogleCloudDialogflowCxV3Webhook', 1, repeated=True)