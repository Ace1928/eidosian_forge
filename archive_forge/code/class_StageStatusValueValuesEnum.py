from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StageStatusValueValuesEnum(_messages.Enum):
    """Optional. List only stages in the given state.

    Values:
      STAGE_STATUS_UNSPECIFIED: <no description>
      STAGE_STATUS_ACTIVE: <no description>
      STAGE_STATUS_COMPLETE: <no description>
      STAGE_STATUS_FAILED: <no description>
      STAGE_STATUS_PENDING: <no description>
      STAGE_STATUS_SKIPPED: <no description>
    """
    STAGE_STATUS_UNSPECIFIED = 0
    STAGE_STATUS_ACTIVE = 1
    STAGE_STATUS_COMPLETE = 2
    STAGE_STATUS_FAILED = 3
    STAGE_STATUS_PENDING = 4
    STAGE_STATUS_SKIPPED = 5