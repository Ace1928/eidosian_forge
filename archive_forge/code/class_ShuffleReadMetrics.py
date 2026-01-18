from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ShuffleReadMetrics(_messages.Message):
    """Shuffle data read by the task.

  Fields:
    fetchWaitTimeMillis: A string attribute.
    localBlocksFetched: A string attribute.
    localBytesRead: A string attribute.
    recordsRead: A string attribute.
    remoteBlocksFetched: A string attribute.
    remoteBytesRead: A string attribute.
    remoteBytesReadToDisk: A string attribute.
    remoteReqsDuration: A string attribute.
    shufflePushReadMetrics: A ShufflePushReadMetrics attribute.
  """
    fetchWaitTimeMillis = _messages.IntegerField(1)
    localBlocksFetched = _messages.IntegerField(2)
    localBytesRead = _messages.IntegerField(3)
    recordsRead = _messages.IntegerField(4)
    remoteBlocksFetched = _messages.IntegerField(5)
    remoteBytesRead = _messages.IntegerField(6)
    remoteBytesReadToDisk = _messages.IntegerField(7)
    remoteReqsDuration = _messages.IntegerField(8)
    shufflePushReadMetrics = _messages.MessageField('ShufflePushReadMetrics', 9)