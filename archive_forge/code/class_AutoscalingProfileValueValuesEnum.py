from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AutoscalingProfileValueValuesEnum(_messages.Enum):
    """Defines autoscaling behaviour.

    Values:
      PROFILE_UNSPECIFIED: No change to autoscaling configuration.
      OPTIMIZE_UTILIZATION: Prioritize optimizing utilization of resources.
      BALANCED: Use default (balanced) autoscaling configuration.
    """
    PROFILE_UNSPECIFIED = 0
    OPTIMIZE_UTILIZATION = 1
    BALANCED = 2