from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RolloutUpdateTypeValueValuesEnum(_messages.Enum):
    """The type of the rollout update.

    Values:
      ROLLOUT_UPDATE_TYPE_UNSPECIFIED: Rollout update type unspecified.
      PENDING: rollout state updated to pending.
      PENDING_RELEASE: Rollout state updated to pending release.
      IN_PROGRESS: Rollout state updated to in progress.
      CANCELLING: Rollout state updated to cancelling.
      CANCELLED: Rollout state updated to cancelled.
      HALTED: Rollout state updated to halted.
      SUCCEEDED: Rollout state updated to succeeded.
      FAILED: Rollout state updated to failed.
      APPROVAL_REQUIRED: Rollout requires approval.
      APPROVED: Rollout has been approved.
      REJECTED: Rollout has been rejected.
      ADVANCE_REQUIRED: Rollout requires advance to the next phase.
      ADVANCED: Rollout has been advanced.
    """
    ROLLOUT_UPDATE_TYPE_UNSPECIFIED = 0
    PENDING = 1
    PENDING_RELEASE = 2
    IN_PROGRESS = 3
    CANCELLING = 4
    CANCELLED = 5
    HALTED = 6
    SUCCEEDED = 7
    FAILED = 8
    APPROVAL_REQUIRED = 9
    APPROVED = 10
    REJECTED = 11
    ADVANCE_REQUIRED = 12
    ADVANCED = 13