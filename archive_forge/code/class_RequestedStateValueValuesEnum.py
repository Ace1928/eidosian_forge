from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RequestedStateValueValuesEnum(_messages.Enum):
    """Optional. The intended state to which the application is reconciling.

    Values:
      STATE_UNSPECIFIED: The application state is unknown.
      PENDING: The application is setting up and has not yet begun to execute
      RUNNING: The application is running.
      CANCELLING: The application is being cancelled.
      CANCELLED: The application was successfully cancelled
      SUCCEEDED: The application completed successfully.
      FAILED: The application exited with an error.
    """
    STATE_UNSPECIFIED = 0
    PENDING = 1
    RUNNING = 2
    CANCELLING = 3
    CANCELLED = 4
    SUCCEEDED = 5
    FAILED = 6