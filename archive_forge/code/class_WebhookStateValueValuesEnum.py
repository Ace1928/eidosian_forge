from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WebhookStateValueValuesEnum(_messages.Enum):
    """Optional. Indicates whether webhooks are enabled for the intent.

    Values:
      WEBHOOK_STATE_UNSPECIFIED: Webhook is disabled in the agent and in the
        intent.
      WEBHOOK_STATE_ENABLED: Webhook is enabled in the agent and in the
        intent.
      WEBHOOK_STATE_ENABLED_FOR_SLOT_FILLING: Webhook is enabled in the agent
        and in the intent. Also, each slot filling prompt is forwarded to the
        webhook.
    """
    WEBHOOK_STATE_UNSPECIFIED = 0
    WEBHOOK_STATE_ENABLED = 1
    WEBHOOK_STATE_ENABLED_FOR_SLOT_FILLING = 2