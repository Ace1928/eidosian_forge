from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StatisticValueValuesEnum(_messages.Enum):
    """Optional. The aggregate metric to evaluate.

    Values:
      STATISTIC_UNDEFINED: Unspecified statistic type
      MEAN: Evaluate the column mean
      MIN: Evaluate the column min
      MAX: Evaluate the column max
    """
    STATISTIC_UNDEFINED = 0
    MEAN = 1
    MIN = 2
    MAX = 3