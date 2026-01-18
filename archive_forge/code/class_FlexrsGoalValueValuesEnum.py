from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FlexrsGoalValueValuesEnum(_messages.Enum):
    """Set FlexRS goal for the job.
    https://cloud.google.com/dataflow/docs/guides/flexrs

    Values:
      FLEXRS_UNSPECIFIED: Run in the default mode.
      FLEXRS_SPEED_OPTIMIZED: Optimize for lower execution time.
      FLEXRS_COST_OPTIMIZED: Optimize for lower cost.
    """
    FLEXRS_UNSPECIFIED = 0
    FLEXRS_SPEED_OPTIMIZED = 1
    FLEXRS_COST_OPTIMIZED = 2