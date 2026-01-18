from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PartToExtractValueValuesEnum(_messages.Enum):
    """The part of the time to keep.

    Values:
      TIME_PART_UNSPECIFIED: Unused
      YEAR: [0-9999]
      MONTH: [1-12]
      DAY_OF_MONTH: [1-31]
      DAY_OF_WEEK: [1-7]
      WEEK_OF_YEAR: [1-53]
      HOUR_OF_DAY: [0-23]
    """
    TIME_PART_UNSPECIFIED = 0
    YEAR = 1
    MONTH = 2
    DAY_OF_MONTH = 3
    DAY_OF_WEEK = 4
    WEEK_OF_YEAR = 5
    HOUR_OF_DAY = 6