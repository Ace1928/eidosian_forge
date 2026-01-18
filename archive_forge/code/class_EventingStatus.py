from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EventingStatus(_messages.Message):
    """EventingStatus indicates the state of eventing.

  Enums:
    StateValueValuesEnum: Output only. State.

  Fields:
    description: Output only. Description of error if State is set to "ERROR".
    state: Output only. State.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State.

    Values:
      STATE_UNSPECIFIED: Default state.
      ACTIVE: Eventing is enabled and ready to receive events.
      ERROR: Eventing is not active due to an error.
    """
        STATE_UNSPECIFIED = 0
        ACTIVE = 1
        ERROR = 2
    description = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)