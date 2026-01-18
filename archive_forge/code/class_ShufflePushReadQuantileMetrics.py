from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShufflePushReadQuantileMetrics(_messages.Message):
    """A ShufflePushReadQuantileMetrics object.

  Fields:
    corruptMergedBlockChunks: A Quantiles attribute.
    localMergedBlocksFetched: A Quantiles attribute.
    localMergedBytesRead: A Quantiles attribute.
    localMergedChunksFetched: A Quantiles attribute.
    mergedFetchFallbackCount: A Quantiles attribute.
    remoteMergedBlocksFetched: A Quantiles attribute.
    remoteMergedBytesRead: A Quantiles attribute.
    remoteMergedChunksFetched: A Quantiles attribute.
    remoteMergedReqsDuration: A Quantiles attribute.
  """
    corruptMergedBlockChunks = _messages.MessageField('Quantiles', 1)
    localMergedBlocksFetched = _messages.MessageField('Quantiles', 2)
    localMergedBytesRead = _messages.MessageField('Quantiles', 3)
    localMergedChunksFetched = _messages.MessageField('Quantiles', 4)
    mergedFetchFallbackCount = _messages.MessageField('Quantiles', 5)
    remoteMergedBlocksFetched = _messages.MessageField('Quantiles', 6)
    remoteMergedBytesRead = _messages.MessageField('Quantiles', 7)
    remoteMergedChunksFetched = _messages.MessageField('Quantiles', 8)
    remoteMergedReqsDuration = _messages.MessageField('Quantiles', 9)