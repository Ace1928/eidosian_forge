from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FlexResourceSchedulingGoalValueValuesEnum(_messages.Enum):
    """Which Flexible Resource Scheduling mode to run in.

    Values:
      FLEXRS_UNSPECIFIED: Run in the default mode.
      FLEXRS_SPEED_OPTIMIZED: Optimize for lower execution time.
      FLEXRS_COST_OPTIMIZED: Optimize for lower cost.
    """
    FLEXRS_UNSPECIFIED = 0
    FLEXRS_SPEED_OPTIMIZED = 1
    FLEXRS_COST_OPTIMIZED = 2