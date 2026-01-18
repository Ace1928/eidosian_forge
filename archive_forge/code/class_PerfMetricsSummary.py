from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PerfMetricsSummary(_messages.Message):
    """A summary of perf metrics collected and performance environment info

  Enums:
    PerfMetricsValueListEntryValuesEnum:

  Fields:
    appStartTime: A AppStartTime attribute.
    executionId: A tool results execution ID. @OutputOnly
    graphicsStats: Graphics statistics for the entire run. Statistics are
      reset at the beginning of the run and collected at the end of the run.
    historyId: A tool results history ID. @OutputOnly
    perfEnvironment: Describes the environment in which the performance
      metrics were collected
    perfMetrics: Set of resource collected
    projectId: The cloud project @OutputOnly
    stepId: A tool results step ID. @OutputOnly
  """

    class PerfMetricsValueListEntryValuesEnum(_messages.Enum):
        """PerfMetricsValueListEntryValuesEnum enum type.

    Values:
      perfMetricTypeUnspecified: <no description>
      memory: <no description>
      cpu: <no description>
      network: <no description>
      graphics: <no description>
    """
        perfMetricTypeUnspecified = 0
        memory = 1
        cpu = 2
        network = 3
        graphics = 4
    appStartTime = _messages.MessageField('AppStartTime', 1)
    executionId = _messages.StringField(2)
    graphicsStats = _messages.MessageField('GraphicsStats', 3)
    historyId = _messages.StringField(4)
    perfEnvironment = _messages.MessageField('PerfEnvironment', 5)
    perfMetrics = _messages.EnumField('PerfMetricsValueListEntryValuesEnum', 6, repeated=True)
    projectId = _messages.StringField(7)
    stepId = _messages.StringField(8)