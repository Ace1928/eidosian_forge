from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubSubDestination(_messages.Message):
    """Specifies the pub/sub destination to send the notifications to.

  Fields:
    topic: Required. A Pub/Sub topic to which messages are sent by GCMA.
      https://cloud.google.com/pubsub/docs/overview
  """
    topic = _messages.StringField(1)