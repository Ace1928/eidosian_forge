from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ExpectedStateValueValuesEnum(_messages.Enum):
    """The state that the instance is expected to be in. For example, an
    instance state can transition to UNHEALTHY due to wrong patch update,
    while the expected state will remain at the HEALTHY.

    Values:
      STATE_UNSPECIFIED: <no description>
      HEALTHY: The instance is running.
      UNHEALTHY: Instance being created, updated, deleted or under maintenance
      SUSPENDED: When instance is suspended
      DELETED: Instance is deleted.
      STATE_OTHER: For rest of the other category
    """
    STATE_UNSPECIFIED = 0
    HEALTHY = 1
    UNHEALTHY = 2
    SUSPENDED = 3
    DELETED = 4
    STATE_OTHER = 5