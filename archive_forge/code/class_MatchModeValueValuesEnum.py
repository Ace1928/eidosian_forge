from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class MatchModeValueValuesEnum(_messages.Enum):
    """Optional. Determines how intents are detected from user queries.

    Values:
      MATCH_MODE_UNSPECIFIED: Not specified.
      MATCH_MODE_HYBRID: Best for agents with a small number of examples in
        intents and/or wide use of templates syntax and composite entities.
      MATCH_MODE_ML_ONLY: Can be used for agents with a large number of
        examples in intents, especially the ones using @sys.any or very large
        custom entities.
    """
    MATCH_MODE_UNSPECIFIED = 0
    MATCH_MODE_HYBRID = 1
    MATCH_MODE_ML_ONLY = 2