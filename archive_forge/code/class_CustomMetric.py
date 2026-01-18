from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CustomMetric(_messages.Message):
    """Allows autoscaling based on Stackdriver metrics.

  Fields:
    filter: Allows filtering on the metric's fields.
    metricName: The name of the metric.
    singleInstanceAssignment: May be used instead of target_utilization when
      an instance can handle a specific amount of work/resources and the
      metric value is equal to the current amount of work remaining. The
      autoscaler will try to keep the number of instances equal to the metric
      value divided by single_instance_assignment.
    targetType: The type of the metric. Must be a string representing a
      Stackdriver metric type e.g. GAGUE, DELTA_PER_SECOND, etc.
    targetUtilization: The target value for the metric.
  """
    filter = _messages.StringField(1)
    metricName = _messages.StringField(2)
    singleInstanceAssignment = _messages.FloatField(3)
    targetType = _messages.StringField(4)
    targetUtilization = _messages.FloatField(5)