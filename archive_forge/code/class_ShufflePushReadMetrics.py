from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShufflePushReadMetrics(_messages.Message):
    """A ShufflePushReadMetrics object.

  Fields:
    corruptMergedBlockChunks: A string attribute.
    localMergedBlocksFetched: A string attribute.
    localMergedBytesRead: A string attribute.
    localMergedChunksFetched: A string attribute.
    mergedFetchFallbackCount: A string attribute.
    remoteMergedBlocksFetched: A string attribute.
    remoteMergedBytesRead: A string attribute.
    remoteMergedChunksFetched: A string attribute.
    remoteMergedReqsDuration: A string attribute.
  """
    corruptMergedBlockChunks = _messages.IntegerField(1)
    localMergedBlocksFetched = _messages.IntegerField(2)
    localMergedBytesRead = _messages.IntegerField(3)
    localMergedChunksFetched = _messages.IntegerField(4)
    mergedFetchFallbackCount = _messages.IntegerField(5)
    remoteMergedBlocksFetched = _messages.IntegerField(6)
    remoteMergedBytesRead = _messages.IntegerField(7)
    remoteMergedChunksFetched = _messages.IntegerField(8)
    remoteMergedReqsDuration = _messages.IntegerField(9)