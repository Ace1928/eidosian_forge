from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SessionStateHistory(_messages.Message):
    """Historical state information.

  Enums:
    StateValueValuesEnum: Output only. The state of the session at this point
      in the session history.

  Fields:
    state: Output only. The state of the session at this point in the session
      history.
    stateMessage: Output only. Details about the state at this point in the
      session history.
    stateStartTime: Output only. The time when the session entered the
      historical state.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. The state of the session at this point in the session
    history.

    Values:
      STATE_UNSPECIFIED: The session state is unknown.
      CREATING: The session is created prior to running.
      ACTIVE: The session is running.
      TERMINATING: The session is terminating.
      TERMINATED: The session is terminated successfully.
      FAILED: The session is no longer running due to an error.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        TERMINATING = 3
        TERMINATED = 4
        FAILED = 5
    state = _messages.EnumField('StateValueValuesEnum', 1)
    stateMessage = _messages.StringField(2)
    stateStartTime = _messages.StringField(3)