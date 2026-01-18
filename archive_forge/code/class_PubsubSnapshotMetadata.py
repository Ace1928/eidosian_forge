from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PubsubSnapshotMetadata(_messages.Message):
    """Represents a Pubsub snapshot.

  Fields:
    expireTime: The expire time of the Pubsub snapshot.
    snapshotName: The name of the Pubsub snapshot.
    topicName: The name of the Pubsub topic.
  """
    expireTime = _messages.StringField(1)
    snapshotName = _messages.StringField(2)
    topicName = _messages.StringField(3)