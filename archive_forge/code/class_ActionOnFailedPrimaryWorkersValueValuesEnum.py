from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActionOnFailedPrimaryWorkersValueValuesEnum(_messages.Enum):
    """Optional. Failure action when primary worker creation fails.

    Values:
      FAILURE_ACTION_UNSPECIFIED: When FailureAction is unspecified, failure
        action defaults to NO_ACTION.
      NO_ACTION: Take no action on failure to create a cluster resource.
        NO_ACTION is the default.
      DELETE: Delete the failed cluster resource.
    """
    FAILURE_ACTION_UNSPECIFIED = 0
    NO_ACTION = 1
    DELETE = 2