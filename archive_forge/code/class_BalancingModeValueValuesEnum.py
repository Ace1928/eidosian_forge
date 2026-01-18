from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BalancingModeValueValuesEnum(_messages.Enum):
    """Specifies how to determine whether the backend of a load balancer can
    handle additional traffic or is fully loaded. For usage guidelines, see
    Connection balancing mode. Backends must use compatible balancing modes.
    For more information, see Supported balancing modes and target capacity
    settings and Restrictions and guidance for instance groups. Note:
    Currently, if you use the API to configure incompatible balancing modes,
    the configuration might be accepted even though it has no impact and is
    ignored. Specifically, Backend.maxUtilization is ignored when
    Backend.balancingMode is RATE. In the future, this incompatible
    combination will be rejected.

    Values:
      CONNECTION: Balance based on the number of simultaneous connections.
      RATE: Balance based on requests per second (RPS).
      UTILIZATION: Balance based on the backend utilization.
    """
    CONNECTION = 0
    RATE = 1
    UTILIZATION = 2