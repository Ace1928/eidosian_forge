from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class AlignmentValueValuesEnum(_messages.Enum):
    """Optional. The alignment of the timed counts to be returned. Default is
    `ALIGNMENT_EQUAL_AT_END`.

    Values:
      ERROR_COUNT_ALIGNMENT_UNSPECIFIED: No alignment specified.
      ALIGNMENT_EQUAL_ROUNDED: The time periods shall be consecutive, have
        width equal to the requested duration, and be aligned at the
        alignment_time provided in the request. The alignment_time does not
        have to be inside the query period but even if it is outside, only
        time periods are returned which overlap with the query period. A
        rounded alignment will typically result in a different size of the
        first or the last time period.
      ALIGNMENT_EQUAL_AT_END: The time periods shall be consecutive, have
        width equal to the requested duration, and be aligned at the end of
        the requested time period. This can result in a different size of the
        first time period.
    """
    ERROR_COUNT_ALIGNMENT_UNSPECIFIED = 0
    ALIGNMENT_EQUAL_ROUNDED = 1
    ALIGNMENT_EQUAL_AT_END = 2