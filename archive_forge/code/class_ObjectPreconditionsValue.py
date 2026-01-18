from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ObjectPreconditionsValue(_messages.Message):
    """Conditions that must be met for this operation to execute.

      Fields:
        ifGenerationMatch: Only perform the composition if the generation of
          the source object that would be used matches this value. If this
          value and a generation are both specified, they must be the same
          value or the call will fail.
      """
    ifGenerationMatch = _messages.IntegerField(1)