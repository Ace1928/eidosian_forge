from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SparkWrapperObject(_messages.Message):
    """Outer message that contains the data obtained from spark listener,
  packaged with information that is required to process it.

  Fields:
    appSummary: A AppSummary attribute.
    applicationEnvironmentInfo: A ApplicationEnvironmentInfo attribute.
    applicationId: Application Id created by Spark.
    applicationInfo: A ApplicationInfo attribute.
    eventTimestamp: VM Timestamp associated with the data object.
    executorStageSummary: A ExecutorStageSummary attribute.
    executorSummary: A ExecutorSummary attribute.
    jobData: A JobData attribute.
    poolData: A PoolData attribute.
    processSummary: A ProcessSummary attribute.
    rddOperationGraph: A RddOperationGraph attribute.
    rddStorageInfo: A RddStorageInfo attribute.
    resourceProfileInfo: A ResourceProfileInfo attribute.
    sparkPlanGraph: A SparkPlanGraph attribute.
    speculationStageSummary: A SpeculationStageSummary attribute.
    sqlExecutionUiData: A SqlExecutionUiData attribute.
    stageData: A StageData attribute.
    streamBlockData: A StreamBlockData attribute.
    streamingQueryData: A StreamingQueryData attribute.
    streamingQueryProgress: A StreamingQueryProgress attribute.
    taskData: A TaskData attribute.
  """
    appSummary = _messages.MessageField('AppSummary', 1)
    applicationEnvironmentInfo = _messages.MessageField('ApplicationEnvironmentInfo', 2)
    applicationId = _messages.StringField(3)
    applicationInfo = _messages.MessageField('ApplicationInfo', 4)
    eventTimestamp = _messages.StringField(5)
    executorStageSummary = _messages.MessageField('ExecutorStageSummary', 6)
    executorSummary = _messages.MessageField('ExecutorSummary', 7)
    jobData = _messages.MessageField('JobData', 8)
    poolData = _messages.MessageField('PoolData', 9)
    processSummary = _messages.MessageField('ProcessSummary', 10)
    rddOperationGraph = _messages.MessageField('RddOperationGraph', 11)
    rddStorageInfo = _messages.MessageField('RddStorageInfo', 12)
    resourceProfileInfo = _messages.MessageField('ResourceProfileInfo', 13)
    sparkPlanGraph = _messages.MessageField('SparkPlanGraph', 14)
    speculationStageSummary = _messages.MessageField('SpeculationStageSummary', 15)
    sqlExecutionUiData = _messages.MessageField('SqlExecutionUiData', 16)
    stageData = _messages.MessageField('StageData', 17)
    streamBlockData = _messages.MessageField('StreamBlockData', 18)
    streamingQueryData = _messages.MessageField('StreamingQueryData', 19)
    streamingQueryProgress = _messages.MessageField('StreamingQueryProgress', 20)
    taskData = _messages.MessageField('TaskData', 21)