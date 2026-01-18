from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RetryAttempt(_messages.Message):
    """RetryAttempt represents an action of retrying the failed Cloud Deploy
  job.

  Enums:
    StateValueValuesEnum: Output only. Valid state of this retry action.

  Fields:
    attempt: Output only. The index of this retry attempt.
    state: Output only. Valid state of this retry action.
    stateDesc: Output only. Description of the state of the Retry.
    wait: Output only. How long the operation will be paused.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. Valid state of this retry action.

    Values:
      REPAIR_STATE_UNSPECIFIED: The `repair` has an unspecified state.
      REPAIR_STATE_SUCCEEDED: The `repair` action has succeeded.
      REPAIR_STATE_CANCELLED: The `repair` action was cancelled.
      REPAIR_STATE_FAILED: The `repair` action has failed.
      REPAIR_STATE_IN_PROGRESS: The `repair` action is in progress.
      REPAIR_STATE_PENDING: The `repair` action is pending.
      REPAIR_STATE_SKIPPED: The `repair` action was skipped.
      REPAIR_STATE_ABORTED: The `repair` action was aborted.
    """
        REPAIR_STATE_UNSPECIFIED = 0
        REPAIR_STATE_SUCCEEDED = 1
        REPAIR_STATE_CANCELLED = 2
        REPAIR_STATE_FAILED = 3
        REPAIR_STATE_IN_PROGRESS = 4
        REPAIR_STATE_PENDING = 5
        REPAIR_STATE_SKIPPED = 6
        REPAIR_STATE_ABORTED = 7
    attempt = _messages.IntegerField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    stateDesc = _messages.StringField(3)
    wait = _messages.StringField(4)