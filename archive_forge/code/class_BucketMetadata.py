from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class BucketMetadata(_messages.Message):
    """Metadata for LongRunningUpdateBucket Operations.

  Enums:
    StateValueValuesEnum: Output only. State of an operation.

  Fields:
    createBucketRequest: LongRunningCreateBucket RPC request.
    endTime: The end time of an operation.
    startTime: The create time of an operation.
    state: Output only. State of an operation.
    updateBucketRequest: LongRunningUpdateBucket RPC request.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. State of an operation.

    Values:
      OPERATION_STATE_UNSPECIFIED: Should not be used.
      OPERATION_STATE_SCHEDULED: The operation is scheduled.
      OPERATION_STATE_WAITING_FOR_PERMISSIONS: Waiting for necessary
        permissions.
      OPERATION_STATE_RUNNING: The operation is running.
      OPERATION_STATE_SUCCEEDED: The operation was completed successfully.
      OPERATION_STATE_FAILED: The operation failed.
      OPERATION_STATE_CANCELLED: The operation was cancelled by the user.
      OPERATION_STATE_PENDING: The operation is waiting for quota.
    """
        OPERATION_STATE_UNSPECIFIED = 0
        OPERATION_STATE_SCHEDULED = 1
        OPERATION_STATE_WAITING_FOR_PERMISSIONS = 2
        OPERATION_STATE_RUNNING = 3
        OPERATION_STATE_SUCCEEDED = 4
        OPERATION_STATE_FAILED = 5
        OPERATION_STATE_CANCELLED = 6
        OPERATION_STATE_PENDING = 7
    createBucketRequest = _messages.MessageField('CreateBucketRequest', 1)
    endTime = _messages.StringField(2)
    startTime = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    updateBucketRequest = _messages.MessageField('UpdateBucketRequest', 5)