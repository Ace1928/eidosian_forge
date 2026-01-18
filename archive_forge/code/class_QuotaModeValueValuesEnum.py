from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaModeValueValuesEnum(_messages.Enum):
    """Quota mode for this operation.

    Values:
      ACQUIRE: Decreases available quota by the cost specified for the
        operation. If cost is higher than available quota, operation fails and
        returns error.
      ACQUIRE_BEST_EFFORT: Decreases available quota by the cost specified for
        the operation. If cost is higher than available quota, operation does
        not fail and available quota goes down to zero but it returns error.
      CHECK: Does not change any available quota. Only checks if there is
        enough quota. No lock is placed on the checked tokens neither.
    """
    ACQUIRE = 0
    ACQUIRE_BEST_EFFORT = 1
    CHECK = 2