from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SlackDelivery(_messages.Message):
    """SlackDelivery is the delivery configuration for delivering Slack
  messages via webhooks. See Slack webhook documentation at:
  https://api.slack.com/messaging/webhooks.

  Fields:
    webhookUri: The secret reference for the Slack webhook URI for sending
      messages to a channel.
  """
    webhookUri = _messages.MessageField('NotifierSecretRef', 1)