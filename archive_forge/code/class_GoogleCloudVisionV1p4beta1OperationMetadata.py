from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudVisionV1p4beta1OperationMetadata(_messages.Message):
    """Contains metadata for the BatchAnnotateImages operation.

  Enums:
    StateValueValuesEnum: Current state of the batch operation.

  Fields:
    createTime: The time when the batch request was received.
    state: Current state of the batch operation.
    updateTime: The time when the operation result was last updated.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Current state of the batch operation.

    Values:
      STATE_UNSPECIFIED: Invalid.
      CREATED: Request is received.
      RUNNING: Request is actively being processed.
      DONE: The batch processing is done.
      CANCELLED: The batch processing was cancelled.
    """
        STATE_UNSPECIFIED = 0
        CREATED = 1
        RUNNING = 2
        DONE = 3
        CANCELLED = 4
    createTime = _messages.StringField(1)
    state = _messages.EnumField('StateValueValuesEnum', 2)
    updateTime = _messages.StringField(3)