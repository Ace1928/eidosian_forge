from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ActionsValueListEntryValuesEnum(_messages.Enum):
    """ActionsValueListEntryValuesEnum enum type.

    Values:
      ACTIONS_UNSPECIFIED: Unspecified.
      ADVANCE: Advance the rollout to the next phase.
      APPROVE: Approve the rollout.
      CANCEL: Cancel the rollout.
      CREATE: Create a rollout.
      IGNORE_JOB: Ignore a job result on the rollout.
      RETRY_JOB: Retry a job for a rollout.
      ROLLBACK: Rollback a rollout.
      TERMINATE_JOBRUN: Terminate a jobrun.
    """
    ACTIONS_UNSPECIFIED = 0
    ADVANCE = 1
    APPROVE = 2
    CANCEL = 3
    CREATE = 4
    IGNORE_JOB = 5
    RETRY_JOB = 6
    ROLLBACK = 7
    TERMINATE_JOBRUN = 8