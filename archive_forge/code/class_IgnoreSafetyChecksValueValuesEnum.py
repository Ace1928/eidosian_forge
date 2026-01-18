from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class IgnoreSafetyChecksValueValuesEnum(_messages.Enum):
    """The list of quota safety checks to be ignored.

    Values:
      QUOTA_SAFETY_CHECK_UNSPECIFIED: Unspecified quota safety check.
      QUOTA_DECREASE_BELOW_USAGE: Validates that a quota mutation would not
        cause the consumer's effective limit to be lower than the consumer's
        quota usage.
      QUOTA_DECREASE_PERCENTAGE_TOO_HIGH: Validates that a quota mutation
        would not cause the consumer's effective limit to decrease by more
        than 10 percent.
    """
    QUOTA_SAFETY_CHECK_UNSPECIFIED = 0
    QUOTA_DECREASE_BELOW_USAGE = 1
    QUOTA_DECREASE_PERCENTAGE_TOO_HIGH = 2