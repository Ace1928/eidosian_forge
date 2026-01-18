from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class PatchJobStateValueValuesEnum(_messages.Enum):
    """State of the overall patch. If the patch is no longer active, the
    agent should not begin a new patch step.

    Values:
      PATCH_JOB_STATE_UNSPECIFIED: Unspecified is invalid.
      ACTIVE: The patch job is running. Instances will continue to run patch
        job steps.
      COMPLETED: The patch job is complete.
    """
    PATCH_JOB_STATE_UNSPECIFIED = 0
    ACTIVE = 1
    COMPLETED = 2