from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RedactLogEntriesMetadata(_messages.Message):
    """Metadata for RedactLogEntries long running operations.

  Enums:
    StateValueValuesEnum: Output only. State of an operation.

  Fields:
    cancellationRequested: Identifies whether the user has requested
      cancellation of the operation.
    endTime: The time at which the operation completed.
    impactAssessment: The expected impact of the operation. If not set, impact
      has not been fully assessed.
    progress: Estimated progress of the operation (0 - 100%).
    receiveTime: The time at which the redaction request was received.
    request: RedactLogEntries RPC request.
    startTime: The time at which redaction of log entries commenced.
    state: Output only. State of an operation.
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
    cancellationRequested = _messages.BooleanField(1)
    endTime = _messages.StringField(2)
    impactAssessment = _messages.MessageField('RedactLogEntriesImpact', 3)
    progress = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    receiveTime = _messages.StringField(5)
    request = _messages.MessageField('RedactLogEntriesRequest', 6)
    startTime = _messages.StringField(7)
    state = _messages.EnumField('StateValueValuesEnum', 8)