from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetentionConfig(_messages.Message):
    """The settings for a topic's message retention.

  Fields:
    perPartitionBytes: The provisioned storage, in bytes, per partition. If
      the number of bytes stored in any of the topic's partitions grows beyond
      this value, older messages will be dropped to make room for newer ones,
      regardless of the value of `period`.
    period: How long a published message is retained. If unset, messages will
      be retained as long as the bytes retained for each partition is below
      `per_partition_bytes`.
  """
    perPartitionBytes = _messages.IntegerField(1)
    period = _messages.StringField(2)