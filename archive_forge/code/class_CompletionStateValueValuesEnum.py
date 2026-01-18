from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CompletionStateValueValuesEnum(_messages.Enum):
    """Output only. Job completion state, i.e. the final state after the job
    completed.

    Values:
      JOB_COMPLETION_STATE_UNSPECIFIED: The status is not specified. This
        state is used when job is not yet finished.
      SUCCEEDED: Success.
      FAILED: Error.
    """
    JOB_COMPLETION_STATE_UNSPECIFIED = 0
    SUCCEEDED = 1
    FAILED = 2