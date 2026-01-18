from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetupErrorsValueListEntryValuesEnum(_messages.Enum):
    """SetupErrorsValueListEntryValuesEnum enum type.

    Values:
      SETUP_ERROR_UNSPECIFIED: Unspecified.
      ERROR_INVALID_BASE_SETUP: Invalid states for all customers, to be
        redirected to AA UI for additional details.
      ERROR_MISSING_EXTERNAL_SIGNING_KEY: Returned when there is not an EKM
        key configured.
      ERROR_NOT_ALL_SERVICES_ENROLLED: Returned when there are no enrolled
        services or the customer is enrolled in CAA only for a subset of
        services.
      ERROR_SETUP_CHECK_FAILED: Returned when exception was encountered during
        evaluation of other criteria.
    """
    SETUP_ERROR_UNSPECIFIED = 0
    ERROR_INVALID_BASE_SETUP = 1
    ERROR_MISSING_EXTERNAL_SIGNING_KEY = 2
    ERROR_NOT_ALL_SERVICES_ENROLLED = 3
    ERROR_SETUP_CHECK_FAILED = 4