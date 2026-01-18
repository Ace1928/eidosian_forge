from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExecutorMetricsDistributions(_messages.Message):
    """A ExecutorMetricsDistributions object.

  Fields:
    diskBytesSpilled: A number attribute.
    failedTasks: A number attribute.
    inputBytes: A number attribute.
    inputRecords: A number attribute.
    killedTasks: A number attribute.
    memoryBytesSpilled: A number attribute.
    outputBytes: A number attribute.
    outputRecords: A number attribute.
    peakMemoryMetrics: A ExecutorPeakMetricsDistributions attribute.
    quantiles: A number attribute.
    shuffleRead: A number attribute.
    shuffleReadRecords: A number attribute.
    shuffleWrite: A number attribute.
    shuffleWriteRecords: A number attribute.
    succeededTasks: A number attribute.
    taskTimeMillis: A number attribute.
  """
    diskBytesSpilled = _messages.FloatField(1, repeated=True)
    failedTasks = _messages.FloatField(2, repeated=True)
    inputBytes = _messages.FloatField(3, repeated=True)
    inputRecords = _messages.FloatField(4, repeated=True)
    killedTasks = _messages.FloatField(5, repeated=True)
    memoryBytesSpilled = _messages.FloatField(6, repeated=True)
    outputBytes = _messages.FloatField(7, repeated=True)
    outputRecords = _messages.FloatField(8, repeated=True)
    peakMemoryMetrics = _messages.MessageField('ExecutorPeakMetricsDistributions', 9)
    quantiles = _messages.FloatField(10, repeated=True)
    shuffleRead = _messages.FloatField(11, repeated=True)
    shuffleReadRecords = _messages.FloatField(12, repeated=True)
    shuffleWrite = _messages.FloatField(13, repeated=True)
    shuffleWriteRecords = _messages.FloatField(14, repeated=True)
    succeededTasks = _messages.FloatField(15, repeated=True)
    taskTimeMillis = _messages.FloatField(16, repeated=True)