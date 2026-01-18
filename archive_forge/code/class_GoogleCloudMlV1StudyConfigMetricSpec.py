from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1StudyConfigMetricSpec(_messages.Message):
    """Represents a metric to optimize.

  Enums:
    GoalValueValuesEnum: Required. The optimization goal of the metric.

  Fields:
    goal: Required. The optimization goal of the metric.
    metric: Required. The name of the metric.
  """

    class GoalValueValuesEnum(_messages.Enum):
        """Required. The optimization goal of the metric.

    Values:
      GOAL_TYPE_UNSPECIFIED: Goal Type will default to maximize.
      MAXIMIZE: Maximize the goal metric.
      MINIMIZE: Minimize the goal metric.
    """
        GOAL_TYPE_UNSPECIFIED = 0
        MAXIMIZE = 1
        MINIMIZE = 2
    goal = _messages.EnumField('GoalValueValuesEnum', 1)
    metric = _messages.StringField(2)