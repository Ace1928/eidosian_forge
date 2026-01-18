from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ErrorStateInfoValueValuesEnum(_messages.Enum):
    """Output only. Additional information on why the subscription is in
    error state.

    Values:
      ERROR_STATE_INFO_UNSPECIFIED: Default value. Should not be used.
      CURRENT_SEGMENT_IS_PENDING: Current segment is pending.
      TERMINATION_SEGMENT_IS_PENDING: Termination segment is pending.
      SWITCH_CASE_FALL_THROUGH: Switch case fall through.
    """
    ERROR_STATE_INFO_UNSPECIFIED = 0
    CURRENT_SEGMENT_IS_PENDING = 1
    TERMINATION_SEGMENT_IS_PENDING = 2
    SWITCH_CASE_FALL_THROUGH = 3