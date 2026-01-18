from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesListRequest(_messages.Message):
    """A ToolresultsProjectsHistoriesExecutionsStepsPerfSampleSeriesListRequest
  object.

  Enums:
    FilterValueValuesEnum: Specify one or more PerfMetricType values such as
      CPU to filter the result

  Fields:
    executionId: A tool results execution ID.
    filter: Specify one or more PerfMetricType values such as CPU to filter
      the result
    historyId: A tool results history ID.
    projectId: The cloud project
    stepId: A tool results step ID.
  """

    class FilterValueValuesEnum(_messages.Enum):
        """Specify one or more PerfMetricType values such as CPU to filter the
    result

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
    executionId = _messages.StringField(1, required=True)
    filter = _messages.EnumField('FilterValueValuesEnum', 2, repeated=True)
    historyId = _messages.StringField(3, required=True)
    projectId = _messages.StringField(4, required=True)
    stepId = _messages.StringField(5, required=True)