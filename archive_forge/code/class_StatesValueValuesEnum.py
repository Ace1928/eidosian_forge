from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StatesValueValuesEnum(_messages.Enum):
    """When specified, only transfer runs with requested states are returned.

    Values:
      TRANSFER_STATE_UNSPECIFIED: State placeholder (0).
      PENDING: Data transfer is scheduled and is waiting to be picked up by
        data transfer backend (2).
      RUNNING: Data transfer is in progress (3).
      SUCCEEDED: Data transfer completed successfully (4).
      FAILED: Data transfer failed (5).
      CANCELLED: Data transfer is cancelled (6).
    """
    TRANSFER_STATE_UNSPECIFIED = 0
    PENDING = 1
    RUNNING = 2
    SUCCEEDED = 3
    FAILED = 4
    CANCELLED = 5