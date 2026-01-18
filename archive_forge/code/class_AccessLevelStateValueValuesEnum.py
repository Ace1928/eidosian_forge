from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessLevelStateValueValuesEnum(_messages.Enum):
    """Evaluation state of an access level

    Values:
      ACCESS_LEVEL_STATE_UNSPECIFIED: Reserved
      ACCESS_LEVEL_STATE_GRANTED: The access level state is granted
      ACCESS_LEVEL_STATE_NOT_GRANTED: The access level state is not granted
      ACCESS_LEVEL_STATE_ERROR: Encounter error when evaluating this access
        level. Note that such error is on the critical path that blocks the
        evaluation; e.g. False || -> ACCESS_LEVEL_STATE_NOT_GRANTED True && ->
        ACCESS_LEVEL_STATE_ERROR
      ACCESS_LEVEL_NOT_EXIST: The access level doesn't exist
    """
    ACCESS_LEVEL_STATE_UNSPECIFIED = 0
    ACCESS_LEVEL_STATE_GRANTED = 1
    ACCESS_LEVEL_STATE_NOT_GRANTED = 2
    ACCESS_LEVEL_STATE_ERROR = 3
    ACCESS_LEVEL_NOT_EXIST = 4