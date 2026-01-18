from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DetailedHealthStateValueValuesEnum(_messages.Enum):
    """[Output Only] The current detailed instance health state.

    Values:
      DRAINING: The instance is being drained. The existing connections to the
        instance have time to complete, but the new ones are being refused.
      HEALTHY: The instance is reachable i.e. a connection to the application
        health checking endpoint can be established, and conforms to the
        requirements defined by the health check.
      TIMEOUT: The instance is unreachable i.e. a connection to the
        application health checking endpoint cannot be established, or the
        server does not respond within the specified timeout.
      UNHEALTHY: The instance is reachable, but does not conform to the
        requirements defined by the health check.
      UNKNOWN: The health checking system is aware of the instance but its
        health is not known at the moment.
    """
    DRAINING = 0
    HEALTHY = 1
    TIMEOUT = 2
    UNHEALTHY = 3
    UNKNOWN = 4