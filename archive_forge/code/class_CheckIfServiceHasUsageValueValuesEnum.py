from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CheckIfServiceHasUsageValueValuesEnum(_messages.Enum):
    """Defines the behavior for checking service usage when disabling a
    service.

    Values:
      CHECK_IF_SERVICE_HAS_USAGE_UNSPECIFIED: When unset, the default behavior
        is used, which is SKIP.
      SKIP: If set, skip checking service usage when disabling a service.
      CHECK: If set, service usage is checked when disabling the service. If a
        service, or its dependents, has usage in the last 30 days, the request
        returns a FAILED_PRECONDITION error.
    """
    CHECK_IF_SERVICE_HAS_USAGE_UNSPECIFIED = 0
    SKIP = 1
    CHECK = 2