from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudMlV1MetricSpec(_messages.Message):
    """MetricSpec contains the specifications to use to calculate the desired
  nodes count when autoscaling is enabled.

  Enums:
    NameValueValuesEnum: metric name.

  Fields:
    name: metric name.
    target: Target specifies the target value for the given metric; once real
      metric deviates from the threshold by a certain percentage, the node
      count changes.
  """

    class NameValueValuesEnum(_messages.Enum):
        """metric name.

    Values:
      METRIC_NAME_UNSPECIFIED: Unspecified MetricName.
      CPU_USAGE: CPU usage.
      GPU_DUTY_CYCLE: GPU duty cycle.
    """
        METRIC_NAME_UNSPECIFIED = 0
        CPU_USAGE = 1
        GPU_DUTY_CYCLE = 2
    name = _messages.EnumField('NameValueValuesEnum', 1)
    target = _messages.IntegerField(2, variant=_messages.Variant.INT32)