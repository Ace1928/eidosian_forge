from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StateNoteValueValuesEnum(_messages.Enum):
    """Output only. An optional field providing information about the current
    instance state.

    Values:
      STATE_NOTE_UNSPECIFIED: STATE_NOTE_UNSPECIFIED as the first value of
        State.
      PAUSED_CMEK_UNAVAILABLE: CMEK access is unavailable.
      INSTANCE_RESUMING: INSTANCE_RESUMING indicates that the instance was
        previously paused and is under the process of being brought back.
    """
    STATE_NOTE_UNSPECIFIED = 0
    PAUSED_CMEK_UNAVAILABLE = 1
    INSTANCE_RESUMING = 2