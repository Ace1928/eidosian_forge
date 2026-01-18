from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class HeadwearLikelihoodValueValuesEnum(_messages.Enum):
    """Headwear likelihood.

    Values:
      UNKNOWN: Unknown likelihood.
      VERY_UNLIKELY: It is very unlikely.
      UNLIKELY: It is unlikely.
      POSSIBLE: It is possible.
      LIKELY: It is likely.
      VERY_LIKELY: It is very likely.
    """
    UNKNOWN = 0
    VERY_UNLIKELY = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    VERY_LIKELY = 5