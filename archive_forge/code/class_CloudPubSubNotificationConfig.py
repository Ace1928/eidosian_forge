from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudPubSubNotificationConfig(_messages.Message):
    """The configuration for Pub/Sub messaging for the connector.

  Fields:
    pubsubSubscription: The Pub/Sub subscription the connector uses to receive
      notifications.
  """
    pubsubSubscription = _messages.StringField(1)