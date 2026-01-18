from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShuffleReadQuantileMetrics(_messages.Message):
    """A ShuffleReadQuantileMetrics object.

  Fields:
    fetchWaitTimeMillis: A Quantiles attribute.
    localBlocksFetched: A Quantiles attribute.
    readBytes: A Quantiles attribute.
    readRecords: A Quantiles attribute.
    remoteBlocksFetched: A Quantiles attribute.
    remoteBytesRead: A Quantiles attribute.
    remoteBytesReadToDisk: A Quantiles attribute.
    remoteReqsDuration: A Quantiles attribute.
    shufflePushReadMetrics: A ShufflePushReadQuantileMetrics attribute.
    totalBlocksFetched: A Quantiles attribute.
  """
    fetchWaitTimeMillis = _messages.MessageField('Quantiles', 1)
    localBlocksFetched = _messages.MessageField('Quantiles', 2)
    readBytes = _messages.MessageField('Quantiles', 3)
    readRecords = _messages.MessageField('Quantiles', 4)
    remoteBlocksFetched = _messages.MessageField('Quantiles', 5)
    remoteBytesRead = _messages.MessageField('Quantiles', 6)
    remoteBytesReadToDisk = _messages.MessageField('Quantiles', 7)
    remoteReqsDuration = _messages.MessageField('Quantiles', 8)
    shufflePushReadMetrics = _messages.MessageField('ShufflePushReadQuantileMetrics', 9)
    totalBlocksFetched = _messages.MessageField('Quantiles', 10)