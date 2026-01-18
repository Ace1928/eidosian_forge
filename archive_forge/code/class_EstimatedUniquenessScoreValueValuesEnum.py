from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EstimatedUniquenessScoreValueValuesEnum(_messages.Enum):
    """Approximate uniqueness of the column.

    Values:
      UNIQUENESS_SCORE_LEVEL_UNSPECIFIED: Some columns do not have estimated
        uniqueness. Possible reasons include having too few values.
      UNIQUENESS_SCORE_LOW: Low uniqueness, possibly a boolean, enum or
        similiarly typed column.
      UNIQUENESS_SCORE_MEDIUM: Medium uniqueness.
      UNIQUENESS_SCORE_HIGH: High uniqueness, possibly a column of free text
        or unique identifiers.
    """
    UNIQUENESS_SCORE_LEVEL_UNSPECIFIED = 0
    UNIQUENESS_SCORE_LOW = 1
    UNIQUENESS_SCORE_MEDIUM = 2
    UNIQUENESS_SCORE_HIGH = 3