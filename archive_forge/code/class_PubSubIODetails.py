from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubSubIODetails(_messages.Message):
    """Metadata for a Pub/Sub connector used by the job.

  Fields:
    subscription: Subscription used in the connection.
    topic: Topic accessed in the connection.
  """
    subscription = _messages.StringField(1)
    topic = _messages.StringField(2)