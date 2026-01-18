from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class VmLivenessValueValuesEnum(_messages.Enum):
    """Output only. The liveness health check of this instance. Only
    applicable for instances in App Engine flexible environment.

    Values:
      LIVENESS_STATE_UNSPECIFIED: There is no liveness health check for the
        instance. Only applicable for instances in App Engine standard
        environment.
      UNKNOWN: The health checking system is aware of the instance but its
        health is not known at the moment.
      HEALTHY: The instance is reachable i.e. a connection to the application
        health checking endpoint can be established, and conforms to the
        requirements defined by the health check.
      UNHEALTHY: The instance is reachable, but does not conform to the
        requirements defined by the health check.
      DRAINING: The instance is being drained. The existing connections to the
        instance have time to complete, but the new ones are being refused.
      TIMEOUT: The instance is unreachable i.e. a connection to the
        application health checking endpoint cannot be established, or the
        server does not respond within the specified timeout.
    """
    LIVENESS_STATE_UNSPECIFIED = 0
    UNKNOWN = 1
    HEALTHY = 2
    UNHEALTHY = 3
    DRAINING = 4
    TIMEOUT = 5