from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1DocumentShardInfo(_messages.Message):
    """For a large document, sharding may be performed to produce several
  document shards. Each document shard contains this field to detail which
  shard it is.

  Fields:
    shardCount: Total number of shards.
    shardIndex: The 0-based index of this shard.
    textOffset: The index of the first character in Document.text in the
      overall document global text.
  """
    shardCount = _messages.IntegerField(1)
    shardIndex = _messages.IntegerField(2)
    textOffset = _messages.IntegerField(3)