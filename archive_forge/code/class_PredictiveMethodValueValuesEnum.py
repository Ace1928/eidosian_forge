from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
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