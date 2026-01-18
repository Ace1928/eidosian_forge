from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthValueValuesEnum(_messages.Enum):
    """The health status of the TPU node.

    Values:
      HEALTH_UNSPECIFIED: Health status is unknown: not initialized or failed
        to retrieve.
      HEALTHY: The resource is healthy.
      TIMEOUT: The resource is unresponsive.
      UNHEALTHY_TENSORFLOW: The in-guest ML stack is unhealthy.
      UNHEALTHY_MAINTENANCE: The node is under maintenance/priority boost
        caused rescheduling and will resume running once rescheduled.
    """
    HEALTH_UNSPECIFIED = 0
    HEALTHY = 1
    TIMEOUT = 2
    UNHEALTHY_TENSORFLOW = 3
    UNHEALTHY_MAINTENANCE = 4