from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Ipv6HealthStateValueValuesEnum(_messages.Enum):
    """Health state of the ipv6 network endpoint determined based on the
    health checks configured.

    Values:
      DRAINING: Endpoint is being drained.
      HEALTHY: Endpoint is healthy.
      UNHEALTHY: Endpoint is unhealthy.
      UNKNOWN: Health status of the endpoint is unknown.
    """
    DRAINING = 0
    HEALTHY = 1
    UNHEALTHY = 2
    UNKNOWN = 3