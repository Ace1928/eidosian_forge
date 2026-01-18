from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MinLikelihoodValueValuesEnum(_messages.Enum):
    """Only returns findings equal to or above this threshold. The default is
    POSSIBLE. In general, the highest likelihood setting yields the fewest
    findings in results and the lowest chance of a false positive. For more
    information, see [Match likelihood](https://cloud.google.com/sensitive-
    data-protection/docs/likelihood).

    Values:
      LIKELIHOOD_UNSPECIFIED: Default value; same as POSSIBLE.
      VERY_UNLIKELY: Highest chance of a false positive.
      UNLIKELY: High chance of a false positive.
      POSSIBLE: Some matching signals. The default value.
      LIKELY: Low chance of a false positive.
      VERY_LIKELY: Confidence level is high. Lowest chance of a false
        positive.
    """
    LIKELIHOOD_UNSPECIFIED = 0
    VERY_UNLIKELY = 1
    UNLIKELY = 2
    POSSIBLE = 3
    LIKELY = 4
    VERY_LIKELY = 5