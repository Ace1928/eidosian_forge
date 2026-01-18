from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AggregationValueValuesEnum(_messages.Enum):
    """The aggregation function used to aggregate each key bucket

    Values:
      AGGREGATION_UNSPECIFIED: Required default value.
      MAX: Use the maximum of all values.
      SUM: Use the sum of all values.
    """
    AGGREGATION_UNSPECIFIED = 0
    MAX = 1
    SUM = 2