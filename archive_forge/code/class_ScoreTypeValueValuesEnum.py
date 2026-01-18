from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ScoreTypeValueValuesEnum(_messages.Enum):
    """ScoreTypeValueValuesEnum enum type.

    Values:
      TYPE_UNKNOWN: Unknown scorer type.
      TYPE_SAFE: Safety scorer.
      TYPE_POLICY: Policy scorer.
      TYPE_GENERATION: Generation scorer.
    """
    TYPE_UNKNOWN = 0
    TYPE_SAFE = 1
    TYPE_POLICY = 2
    TYPE_GENERATION = 3