from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MessageSubscriptionDestination(_messages.Message):
    """Specifications of a message route destination. (For MI MVP it will
  always contain a pubsub transport with a pubsub topic.)

  Fields:
    pubsubTransport: Details of a pubsub transport.
    uri: Required. URI of the destination service to deliver the message to.
      If the destination service name is specified in the incoming request,
      it's considered a point-to-point message and it will only be delivered
      to the route that has the matching destination URI.
  """
    pubsubTransport = _messages.MessageField('MessageSubscriptionPubsubTransport', 1)
    uri = _messages.StringField(2)