from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskQuantileMetrics(_messages.Message):
    """A TaskQuantileMetrics object.

  Fields:
    diskBytesSpilled: A Quantiles attribute.
    durationMillis: A Quantiles attribute.
    executorCpuTimeNanos: A Quantiles attribute.
    executorDeserializeCpuTimeNanos: A Quantiles attribute.
    executorDeserializeTimeMillis: A Quantiles attribute.
    executorRunTimeMillis: A Quantiles attribute.
    gettingResultTimeMillis: A Quantiles attribute.
    inputMetrics: A InputQuantileMetrics attribute.
    jvmGcTimeMillis: A Quantiles attribute.
    memoryBytesSpilled: A Quantiles attribute.
    outputMetrics: A OutputQuantileMetrics attribute.
    peakExecutionMemoryBytes: A Quantiles attribute.
    resultSerializationTimeMillis: A Quantiles attribute.
    resultSize: A Quantiles attribute.
    schedulerDelayMillis: A Quantiles attribute.
    shuffleReadMetrics: A ShuffleReadQuantileMetrics attribute.
    shuffleWriteMetrics: A ShuffleWriteQuantileMetrics attribute.
  """
    diskBytesSpilled = _messages.MessageField('Quantiles', 1)
    durationMillis = _messages.MessageField('Quantiles', 2)
    executorCpuTimeNanos = _messages.MessageField('Quantiles', 3)
    executorDeserializeCpuTimeNanos = _messages.MessageField('Quantiles', 4)
    executorDeserializeTimeMillis = _messages.MessageField('Quantiles', 5)
    executorRunTimeMillis = _messages.MessageField('Quantiles', 6)
    gettingResultTimeMillis = _messages.MessageField('Quantiles', 7)
    inputMetrics = _messages.MessageField('InputQuantileMetrics', 8)
    jvmGcTimeMillis = _messages.MessageField('Quantiles', 9)
    memoryBytesSpilled = _messages.MessageField('Quantiles', 10)
    outputMetrics = _messages.MessageField('OutputQuantileMetrics', 11)
    peakExecutionMemoryBytes = _messages.MessageField('Quantiles', 12)
    resultSerializationTimeMillis = _messages.MessageField('Quantiles', 13)
    resultSize = _messages.MessageField('Quantiles', 14)
    schedulerDelayMillis = _messages.MessageField('Quantiles', 15)
    shuffleReadMetrics = _messages.MessageField('ShuffleReadQuantileMetrics', 16)
    shuffleWriteMetrics = _messages.MessageField('ShuffleWriteQuantileMetrics', 17)