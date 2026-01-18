from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotificationDestination(_messages.Message):
    """Specifies the destination to send the notifications to.

  Fields:
    pubsub: Specifies the pub/sub destination notifications should be sent to.
  """
    pubsub = _messages.MessageField('PubSubDestination', 1)