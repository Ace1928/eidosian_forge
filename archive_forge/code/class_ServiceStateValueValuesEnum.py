from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceStateValueValuesEnum(_messages.Enum):
    """Required. internal service state.

    Values:
      UNKNOWN: Health status is unknown: not initialized or failed to
        retrieve.
      HEALTHY: The resource is healthy.
      UNHEALTHY: The resource is unhealthy.
    """
    UNKNOWN = 0
    HEALTHY = 1
    UNHEALTHY = 2