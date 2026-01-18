from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessageSubscriptionPubsubTransport(_messages.Message):
    """Details of a pubsub transport.

  Fields:
    topic: Required. The pubsub topic to deliver the message to. It should
      match pattern `projects/*/topics/`.
  """
    topic = _messages.StringField(1)