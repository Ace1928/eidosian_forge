from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalingPolicyLoadBalancingUtilization(_messages.Message):
    """Configuration parameters of autoscaling based on load balancing.

  Fields:
    utilizationTarget: Fraction of backend capacity utilization (set in
      HTTP(S) load balancing configuration) that the autoscaler maintains.
      Must be a positive float value. If not defined, the default is 0.8.
  """
    utilizationTarget = _messages.FloatField(1)