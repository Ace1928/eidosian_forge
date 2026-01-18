from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TopicPartitions(_messages.Message):
    """Response for GetTopicPartitions.

  Fields:
    partitionCount: The number of partitions in the topic.
  """
    partitionCount = _messages.IntegerField(1)