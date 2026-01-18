from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class RolloutStateValueValuesEnum(_messages.Enum):
    """State of the rollout

    Values:
      ROLLOUT_STATE_UNSPECIFIED: Invalid value
      IN_PROGRESS: The rollout is in progress.
      CANCELLING: The rollout is being cancelled.
      CANCELLED: The rollout is cancelled.
      SUCCEEDED: The rollout has completed successfully.
    """
    ROLLOUT_STATE_UNSPECIFIED = 0
    IN_PROGRESS = 1
    CANCELLING = 2
    CANCELLED = 3
    SUCCEEDED = 4