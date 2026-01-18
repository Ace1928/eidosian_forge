from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class SetCommonInstanceMetadataOperationMetadataPerLocationOperationInfo(_messages.Message):
    """A SetCommonInstanceMetadataOperationMetadataPerLocationOperationInfo
  object.

  Enums:
    StateValueValuesEnum: [Output Only] Status of the action, which can be one
      of the following: `PROPAGATING`, `PROPAGATED`, `ABANDONED`, `FAILED`, or
      `DONE`.

  Fields:
    error: [Output Only] If state is `ABANDONED` or `FAILED`, this field is
      populated.
    state: [Output Only] Status of the action, which can be one of the
      following: `PROPAGATING`, `PROPAGATED`, `ABANDONED`, `FAILED`, or
      `DONE`.
  """

    class StateValueValuesEnum(_messages.Enum):
        """[Output Only] Status of the action, which can be one of the following:
    `PROPAGATING`, `PROPAGATED`, `ABANDONED`, `FAILED`, or `DONE`.

    Values:
      UNSPECIFIED: <no description>
      PROPAGATING: Operation is not yet confirmed to have been created in the
        location.
      PROPAGATED: Operation is confirmed to be in the location.
      ABANDONED: Operation not tracked in this location e.g. zone is marked as
        DOWN.
      FAILED: Operation is in an error state.
      DONE: Operation has completed successfully.
    """
        UNSPECIFIED = 0
        PROPAGATING = 1
        PROPAGATED = 2
        ABANDONED = 3
        FAILED = 4
        DONE = 5
    error = _messages.MessageField('Status', 1)
    state = _messages.EnumField('StateValueValuesEnum', 2)