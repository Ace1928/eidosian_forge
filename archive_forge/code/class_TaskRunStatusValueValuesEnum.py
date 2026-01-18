from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TaskRunStatusValueValuesEnum(_messages.Enum):
    """Taskrun status the user can provide. Used for cancellation.

    Values:
      TASK_RUN_STATUS_UNSPECIFIED: Default enum type; should not be used.
      TASK_RUN_CANCELLED: Cancelled status.
    """
    TASK_RUN_STATUS_UNSPECIFIED = 0
    TASK_RUN_CANCELLED = 1