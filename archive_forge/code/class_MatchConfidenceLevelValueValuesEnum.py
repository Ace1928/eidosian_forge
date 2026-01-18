from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MatchConfidenceLevelValueValuesEnum(_messages.Enum):
    """The system's confidence level that this knowledge answer is a good
    match for this conversational query. NOTE: The confidence level for a
    given `` pair may change without notice, as it depends on models that are
    constantly being improved. However, it will change less frequently than
    the confidence score below, and should be preferred for referencing the
    quality of an answer.

    Values:
      MATCH_CONFIDENCE_LEVEL_UNSPECIFIED: Not specified.
      LOW: Indicates that the confidence is low.
      MEDIUM: Indicates our confidence is medium.
      HIGH: Indicates our confidence is high.
    """
    MATCH_CONFIDENCE_LEVEL_UNSPECIFIED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3