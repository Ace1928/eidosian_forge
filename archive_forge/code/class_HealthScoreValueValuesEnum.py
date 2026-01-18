from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HealthScoreValueValuesEnum(_messages.Enum):
    """The Health score of the resource. The Health score is the callers
    specification of the condition of the device from a usability point of
    view. For example, a third-party device management provider may specify a
    health score based on its compliance with organizational policies.

    Values:
      HEALTH_SCORE_UNSPECIFIED: Default value
      VERY_POOR: The object is in very poor health as defined by the caller.
      POOR: The object is in poor health as defined by the caller.
      NEUTRAL: The object health is neither good nor poor, as defined by the
        caller.
      GOOD: The object is in good health as defined by the caller.
      VERY_GOOD: The object is in very good health as defined by the caller.
    """
    HEALTH_SCORE_UNSPECIFIED = 0
    VERY_POOR = 1
    POOR = 2
    NEUTRAL = 3
    GOOD = 4
    VERY_GOOD = 5