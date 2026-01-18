from __future__ import annotations
from typing import MutableMapping, MutableSequence
import proto  # type: ignore
from cloudsdk.google.protobuf import field_mask_pb2  # type: ignore
from cloudsdk.google.protobuf import timestamp_pb2  # type: ignore
class OperationState(proto.Enum):
    """List of different operation states.
    High level state of the operation. This is used to report the
    job's current state to the user. Once a long running operation
    is created, the current state of the operation can be queried
    even before the operation is finished and the final result is
    available.

    Values:
        OPERATION_STATE_UNSPECIFIED (0):
            Should not be used.
        OPERATION_STATE_SCHEDULED (1):
            The operation is scheduled.
        OPERATION_STATE_WAITING_FOR_PERMISSIONS (2):
            Waiting for necessary permissions.
        OPERATION_STATE_RUNNING (3):
            The operation is running.
        OPERATION_STATE_SUCCEEDED (4):
            The operation was completed successfully.
        OPERATION_STATE_FAILED (5):
            The operation failed.
        OPERATION_STATE_CANCELLED (6):
            The operation was cancelled by the user.
        OPERATION_STATE_PENDING (7):
            The operation is waiting for quota.
    """
    OPERATION_STATE_UNSPECIFIED = 0
    OPERATION_STATE_SCHEDULED = 1
    OPERATION_STATE_WAITING_FOR_PERMISSIONS = 2
    OPERATION_STATE_RUNNING = 3
    OPERATION_STATE_SUCCEEDED = 4
    OPERATION_STATE_FAILED = 5
    OPERATION_STATE_CANCELLED = 6
    OPERATION_STATE_PENDING = 7