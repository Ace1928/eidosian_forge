from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskData(_messages.Message):
    """Data corresponding to tasks created by spark.

  Messages:
    ExecutorLogsValue: A ExecutorLogsValue object.

  Fields:
    accumulatorUpdates: A AccumulableInfo attribute.
    attempt: A integer attribute.
    durationMillis: A string attribute.
    errorMessage: A string attribute.
    executorId: A string attribute.
    executorLogs: A ExecutorLogsValue attribute.
    gettingResultTimeMillis: A string attribute.
    hasMetrics: A boolean attribute.
    host: A string attribute.
    index: A integer attribute.
    launchTime: A string attribute.
    partitionId: A integer attribute.
    resultFetchStart: A string attribute.
    schedulerDelayMillis: A string attribute.
    speculative: A boolean attribute.
    stageAttemptId: A integer attribute.
    stageId: A string attribute.
    status: A string attribute.
    taskId: A string attribute.
    taskLocality: A string attribute.
    taskMetrics: A TaskMetrics attribute.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class ExecutorLogsValue(_messages.Message):
        """A ExecutorLogsValue object.

    Messages:
      AdditionalProperty: An additional property for a ExecutorLogsValue
        object.

    Fields:
      additionalProperties: Additional properties of type ExecutorLogsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a ExecutorLogsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    accumulatorUpdates = _messages.MessageField('AccumulableInfo', 1, repeated=True)
    attempt = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    durationMillis = _messages.IntegerField(3)
    errorMessage = _messages.StringField(4)
    executorId = _messages.StringField(5)
    executorLogs = _messages.MessageField('ExecutorLogsValue', 6)
    gettingResultTimeMillis = _messages.IntegerField(7)
    hasMetrics = _messages.BooleanField(8)
    host = _messages.StringField(9)
    index = _messages.IntegerField(10, variant=_messages.Variant.INT32)
    launchTime = _messages.StringField(11)
    partitionId = _messages.IntegerField(12, variant=_messages.Variant.INT32)
    resultFetchStart = _messages.StringField(13)
    schedulerDelayMillis = _messages.IntegerField(14)
    speculative = _messages.BooleanField(15)
    stageAttemptId = _messages.IntegerField(16, variant=_messages.Variant.INT32)
    stageId = _messages.IntegerField(17)
    status = _messages.StringField(18)
    taskId = _messages.IntegerField(19)
    taskLocality = _messages.StringField(20)
    taskMetrics = _messages.MessageField('TaskMetrics', 21)