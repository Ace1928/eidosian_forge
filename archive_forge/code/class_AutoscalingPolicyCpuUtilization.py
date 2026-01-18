from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalingPolicyCpuUtilization(_messages.Message):
    """CPU utilization policy.

  Enums:
    PredictiveMethodValueValuesEnum: Indicates whether predictive autoscaling
      based on CPU metric is enabled. Valid values are: * NONE (default). No
      predictive method is used. The autoscaler scales the group to meet
      current demand based on real-time metrics. * OPTIMIZE_AVAILABILITY.
      Predictive autoscaling improves availability by monitoring daily and
      weekly load patterns and scaling out ahead of anticipated demand.

  Fields:
    predictiveMethod: Indicates whether predictive autoscaling based on CPU
      metric is enabled. Valid values are: * NONE (default). No predictive
      method is used. The autoscaler scales the group to meet current demand
      based on real-time metrics. * OPTIMIZE_AVAILABILITY. Predictive
      autoscaling improves availability by monitoring daily and weekly load
      patterns and scaling out ahead of anticipated demand.
    utilizationTarget: The target CPU utilization that the autoscaler
      maintains. Must be a float value in the range (0, 1]. If not specified,
      the default is 0.6. If the CPU level is below the target utilization,
      the autoscaler scales in the number of instances until it reaches the
      minimum number of instances you specified or until the average CPU of
      your instances reaches the target utilization. If the average CPU is
      above the target utilization, the autoscaler scales out until it reaches
      the maximum number of instances you specified or until the average
      utilization reaches the target utilization.
  """

    class PredictiveMethodValueValuesEnum(_messages.Enum):
        """Indicates whether predictive autoscaling based on CPU metric is
    enabled. Valid values are: * NONE (default). No predictive method is used.
    The autoscaler scales the group to meet current demand based on real-time
    metrics. * OPTIMIZE_AVAILABILITY. Predictive autoscaling improves
    availability by monitoring daily and weekly load patterns and scaling out
    ahead of anticipated demand.

    Values:
      NONE: No predictive method is used. The autoscaler scales the group to
        meet current demand based on real-time metrics
      OPTIMIZE_AVAILABILITY: Predictive autoscaling improves availability by
        monitoring daily and weekly load patterns and scaling out ahead of
        anticipated demand.
      PREDICTIVE_METHOD_UNSPECIFIED: <no description>
    """
        NONE = 0
        OPTIMIZE_AVAILABILITY = 1
        PREDICTIVE_METHOD_UNSPECIFIED = 2
    predictiveMethod = _messages.EnumField('PredictiveMethodValueValuesEnum', 1)
    utilizationTarget = _messages.FloatField(2)