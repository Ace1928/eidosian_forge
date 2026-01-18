from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class SessionStateValueValuesEnum(_messages.Enum):
    """Output only. The session_state tracked by this event

    Values:
      SESSION_STATE_UNSPECIFIED: Default value. This value is unused.
      REQUESTED: Initial state of a session request. The session is being
        validated for correctness and a device is not yet requested.
      PENDING: The session has been validated and is in the queue for a
        device.
      ACTIVE: The session has been granted and the device is accepting
        connections.
      EXPIRED: The session duration exceeded the device's reservation time
        period and timed out automatically.
      FINISHED: The user is finished with the session and it was canceled by
        the user while the request was still getting allocated or after
        allocation and during device usage period.
      UNAVAILABLE: Unable to complete the session because the device was
        unavailable and it failed to allocate through the scheduler. For
        example, a device not in the catalog was requested or the request
        expired in the allocation queue.
      ERROR: Unable to complete the session for an internal reason, such as an
        infrastructure failure.
    """
    SESSION_STATE_UNSPECIFIED = 0
    REQUESTED = 1
    PENDING = 2
    ACTIVE = 3
    EXPIRED = 4
    FINISHED = 5
    UNAVAILABLE = 6
    ERROR = 7