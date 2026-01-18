from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TopLevelAccessLevelsEvalStateValueValuesEnum(_messages.Enum):
    """The overall evaluation state of the top level access levels

    Values:
      TOP_LEVEL_ACCESS_LEVELS_EVAL_STATE_UNSPECIFIED: Not used
      NOT_APPLICABLE: The overall evaluation state of the top level access
        levels is not applicable
      GRANTED: The overall evaluation state of the top level access levels is
        granted
      DENIED: The overall evaluation state of the top level access levels is
        denied
    """
    TOP_LEVEL_ACCESS_LEVELS_EVAL_STATE_UNSPECIFIED = 0
    NOT_APPLICABLE = 1
    GRANTED = 2
    DENIED = 3