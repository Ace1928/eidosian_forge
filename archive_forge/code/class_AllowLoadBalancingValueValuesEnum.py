from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AllowLoadBalancingValueValuesEnum(_messages.Enum):
    """Specifies whether cloud load balancing is allowed.

    Values:
      LOAD_BALANCING_ALLOWED: <no description>
      LOAD_BALANCING_BLOCKED: <no description>
      LOAD_BALANCING_UNSPECIFIED: <no description>
    """
    LOAD_BALANCING_ALLOWED = 0
    LOAD_BALANCING_BLOCKED = 1
    LOAD_BALANCING_UNSPECIFIED = 2