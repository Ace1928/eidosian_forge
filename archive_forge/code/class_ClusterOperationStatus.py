from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ClusterOperationStatus(_messages.Message):
    """The status of the operation.

  Enums:
    StateValueValuesEnum: Output only. A message containing the operation
      state.

  Fields:
    details: Output only. A message containing any operation metadata details.
    innerState: Output only. A message containing the detailed operation
      state.
    state: Output only. A message containing the operation state.
    stateStartTime: Output only. The time this state was entered.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. A message containing the operation state.

    Values:
      UNKNOWN: Unused.
      PENDING: The operation has been created.
      RUNNING: The operation is running.
      DONE: The operation is done; either cancelled or completed.
    """
        UNKNOWN = 0
        PENDING = 1
        RUNNING = 2
        DONE = 3
    details = _messages.StringField(1)
    innerState = _messages.StringField(2)
    state = _messages.EnumField('StateValueValuesEnum', 3)
    stateStartTime = _messages.StringField(4)