from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ComparisonValueValuesEnum(_messages.Enum):
    """The comparison to apply between the time series (indicated by filter
    and aggregation) and the threshold (indicated by threshold_value). The
    comparison is applied on each time series, with the time series on the
    left-hand side and the threshold on the right-hand side.Only COMPARISON_LT
    and COMPARISON_GT are supported currently.

    Values:
      COMPARISON_UNSPECIFIED: No ordering relationship is specified.
      COMPARISON_GT: True if the left argument is greater than the right
        argument.
      COMPARISON_GE: True if the left argument is greater than or equal to the
        right argument.
      COMPARISON_LT: True if the left argument is less than the right
        argument.
      COMPARISON_LE: True if the left argument is less than or equal to the
        right argument.
      COMPARISON_EQ: True if the left argument is equal to the right argument.
      COMPARISON_NE: True if the left argument is not equal to the right
        argument.
    """
    COMPARISON_UNSPECIFIED = 0
    COMPARISON_GT = 1
    COMPARISON_GE = 2
    COMPARISON_LT = 3
    COMPARISON_LE = 4
    COMPARISON_EQ = 5
    COMPARISON_NE = 6