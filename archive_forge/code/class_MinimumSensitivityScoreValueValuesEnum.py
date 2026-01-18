from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MinimumSensitivityScoreValueValuesEnum(_messages.Enum):
    """The minimum sensitivity level that triggers the condition.

    Values:
      PROFILE_SCORE_BUCKET_UNSPECIFIED: Unused.
      HIGH: High risk/sensitivity detected.
      MEDIUM_OR_HIGH: Medium or high risk/sensitivity detected.
    """
    PROFILE_SCORE_BUCKET_UNSPECIFIED = 0
    HIGH = 1
    MEDIUM_OR_HIGH = 2